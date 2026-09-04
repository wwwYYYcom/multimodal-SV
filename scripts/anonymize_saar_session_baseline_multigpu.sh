#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
MONITOR_SECONDS="${MONITOR_SECONDS:-60}"
MAX_WORKER_ATTEMPTS="${MAX_WORKER_ATTEMPTS:-20}"
ALLOW_CONCURRENT_GPU_JOB="${ALLOW_CONCURRENT_GPU_JOB:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESERVE_GIB="${RESERVE_GIB:-12}"
FULL_COUNT=66712
PROJECTED_OUTPUT_BYTES=5200000000
RESERVE_BYTES=$((RESERVE_GIB * 1024 * 1024 * 1024))

cd "$PROJECT_ROOT"
PLAN="artifacts/saar/session_baseline/manifests/session_baseline_anonymization_plan.csv"
FINAL_MANIFEST="artifacts/saar/session_baseline/manifests/session_baseline_anonymized_manifest.csv"
TRIALS_DIR="artifacts/saar/session_baseline/manifests"
STREAMVOICE_ROOT="third_party/StreamVoiceAnon"
RUN_ROOT="results/runs/saar_session_baseline"
CHECKPOINT="results/runs/audio_corrected_p1/last.pt"
ORIGINAL_EMBEDDINGS="artifacts/embeddings/original_evaluation_corrected.npz"
ANONYMIZED_EMBEDDINGS="artifacts/saar/session_baseline/embeddings/anonymized_evaluation_corrected.npz"
OUTPUT_ROOT="artifacts/saar/session_baseline"
AUDIO_ROOT="artifacts/saar/anonymized/session_baseline_evaluation"
CONTROL_SUMMARY="artifacts/saar/utterance_random_control/metrics/privacy_summary.csv"
STAMP="$(date +%Y%m%d_%H%M%S_%3N)"
RUN_DIR="$RUN_ROOT/multigpu_${STAMP}"
mkdir -p "$RUN_DIR" "$(dirname "$ANONYMIZED_EMBEDDINGS")" "$AUDIO_ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for required in \
    "$PLAN" "${PLAN%.csv}.audit.json" "$TRIALS_DIR/all_seeds.jsonl" \
    "$CHECKPOINT" "$ORIGINAL_EMBEDDINGS"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done

"$PYTHON_EXE" - "$PLAN" "${PLAN%.csv}.audit.json" <<'PY'
import csv
import json
import sys
from pathlib import Path, PureWindowsPath

plan = Path(sys.argv[1])
audit = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
with plan.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 66712, len(rows)
assert audit["source_utterances"] == 66712
assert audit["mapping"] == "per_session_deterministic_random"
assert audit["trial_role"] == "target"
for row in rows:
    for column in ("audio_path", "reference_audio_path", "output_audio_path"):
        value = row[column]
        assert not PureWindowsPath(value).drive, f"Windows path remains in {column}: {value}"
print("saar_plan_preflight=true")
PY

if [[ "$DRY_RUN" != "1" && "$ALLOW_CONCURRENT_GPU_JOB" != "1" ]] && {
   pgrep -f '[m]msv.cli (anonymize-streamvoice|train-audio|extract-embeddings)' >/dev/null ||
   pgrep -f '[a]nonymize_train_multigpu_then_train_semi.sh' >/dev/null
}; then
    echo "another MMSV GPU job is active; refusing GPU contention" >&2
    echo "wait for the original reproduction, or explicitly set ALLOW_CONCURRENT_GPU_JOB=1" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS_CSV"
if (( ${#GPU_ARRAY[@]} == 0 || WORKERS_PER_GPU <= 0 )); then
    echo "GPU_IDS and WORKERS_PER_GPU must define at least one worker" >&2
    exit 2
fi
TOTAL_WORKERS=$(( ${#GPU_ARRAY[@]} * WORKERS_PER_GPU ))

EXISTING_OUTPUT_BYTES="$(find "$AUDIO_ROOT" -type f -name '*.flac' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s+0}')"
REMAINING_OUTPUT_BYTES=$(( PROJECTED_OUTPUT_BYTES - EXISTING_OUTPUT_BYTES ))
(( REMAINING_OUTPUT_BYTES < 0 )) && REMAINING_OUTPUT_BYTES=0
FREE_BYTES="$(df -PB1 "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
REQUIRED_BYTES=$(( REMAINING_OUTPUT_BYTES + RESERVE_BYTES ))
if (( FREE_BYTES < REQUIRED_BYTES )); then
    echo "insufficient disk space: free=$FREE_BYTES required=$REQUIRED_BYTES" >&2
    exit 2
fi

declare -a STARTS LIMITS GPUS MANIFESTS PIDS ATTEMPTS COMPLETED
BASE=$(( FULL_COUNT / TOTAL_WORKERS ))
REMAINDER=$(( FULL_COUNT % TOTAL_WORKERS ))
cursor=0
for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
    limit=$BASE
    (( worker < REMAINDER )) && limit=$((limit + 1))
    gpu_slot=$(( worker / WORKERS_PER_GPU ))
    STARTS[$worker]=$cursor
    LIMITS[$worker]=$limit
    GPUS[$worker]="${GPU_ARRAY[$gpu_slot]}"
    MANIFESTS[$worker]="$RUN_DIR/worker$((worker + 1)).manifest.csv"
    ATTEMPTS[$worker]=0
    COMPLETED[$worker]=0
    cursor=$((cursor + limit))
done
(( cursor == FULL_COUNT )) || { echo "slice error: $cursor != $FULL_COUNT" >&2; exit 2; }

timestamp() { date '+%Y-%m-%d %H:%M:%S %:z'; }

start_worker() {
    local worker="$1" number attempt manifest progress gpu
    number=$((worker + 1))
    ATTEMPTS[$worker]=$((ATTEMPTS[$worker] + 1))
    attempt="${ATTEMPTS[$worker]}"
    manifest="${MANIFESTS[$worker]}"
    progress="${manifest%.csv}.progress.jsonl"
    gpu="${GPUS[$worker]}"
    if (( attempt > 1 )) && [[ -f "$progress" ]]; then
        mv -f "$progress" "$RUN_DIR/worker${number}.attempt$((attempt - 1)).progress.jsonl"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_EXE" -u -m mmsv.cli anonymize-streamvoice \
        --plan "$PLAN" --output-manifest "$manifest" \
        --streamvoice-root "$STREAMVOICE_ROOT" \
        --delay 2 --alpha 1.0 --max-source-chunk-seconds 30.0 \
        --start-index "${STARTS[$worker]}" --limit "${LIMITS[$worker]}" \
        >"$RUN_DIR/worker${number}.attempt${attempt}.stdout.log" \
        2>"$RUN_DIR/worker${number}.attempt${attempt}.stderr.log" &
    PIDS[$worker]=$!
    echo "worker_started=$(timestamp) worker=$number gpu=$gpu attempt=$attempt pid=${PIDS[$worker]} start=${STARTS[$worker]} limit=${LIMITS[$worker]}"
}

stop_children() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap stop_children INT TERM EXIT

echo "pipeline_started=$(timestamp)"
echo "project_root=$PROJECT_ROOT"
echo "run_dir=$RUN_DIR"
echo "gpu_ids=$GPU_IDS_CSV workers_per_gpu=$WORKERS_PER_GPU total_workers=$TOTAL_WORKERS"
echo "existing_output_bytes=$EXISTING_OUTPUT_BYTES startup_free_bytes=$FREE_BYTES"
if [[ "$DRY_RUN" == "1" ]]; then
    for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
        echo "slice worker=$((worker + 1)) gpu=${GPUS[$worker]} start=${STARTS[$worker]} limit=${LIMITS[$worker]}"
    done
    echo "dry_run_complete=true"
    trap - INT TERM EXIT
    exit 0
fi

ANON_STARTED_EPOCH="$(date +%s)"
for ((worker=0; worker<TOTAL_WORKERS; worker++)); do start_worker "$worker"; done
next_report=0
while :; do
    incomplete=0
    for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
        (( COMPLETED[$worker] == 1 )) && continue
        incomplete=$((incomplete + 1))
        pid="${PIDS[$worker]}"
        kill -0 "$pid" 2>/dev/null && continue
        exit_code=0
        wait "$pid" || exit_code=$?
        audit="${MANIFESTS[$worker]%.csv}.audit.json"
        if [[ -f "$audit" ]] && "$PYTHON_EXE" -c 'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if a["processed"] == int(sys.argv[2]) else 1)' "$audit" "${LIMITS[$worker]}"; then
            COMPLETED[$worker]=1
            incomplete=$((incomplete - 1))
            echo "worker_completed=$(timestamp) worker=$((worker + 1)) gpu=${GPUS[$worker]} attempt=${ATTEMPTS[$worker]}"
        else
            echo "worker_failed=$(timestamp) worker=$((worker + 1)) gpu=${GPUS[$worker]} attempt=${ATTEMPTS[$worker]} exit_code=$exit_code"
            (( ATTEMPTS[$worker] < MAX_WORKER_ATTEMPTS )) || exit 1
            sleep 10
            start_worker "$worker"
        fi
    done
    (( incomplete == 0 )) && break
    now="$(date +%s)"
    if (( now >= next_report )); then
        count="$(find "$AUDIO_ROOT" -type f -name '*.flac' 2>/dev/null | wc -l)"
        "$PYTHON_EXE" - "$count" "$FULL_COUNT" <<'PY'
import sys
done, total = map(int, sys.argv[1:])
print(f"overall_progress={done}/{total} ({done / total * 100:.6f}%) remaining={total-done}")
PY
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits || true
        next_report=$((now + MONITOR_SECONDS))
    fi
    sleep 5
done

ANON_WALL_SECONDS=$(( $(date +%s) - ANON_STARTED_EPOCH ))
merge_args=()
for manifest in "${MANIFESTS[@]}"; do merge_args+=("$manifest"); done
"$PYTHON_EXE" scripts/merge_anonymization_manifests.py \
    --plan "$PLAN" --manifests "${merge_args[@]}" --output "$FINAL_MANIFEST"
"$PYTHON_EXE" scripts/validate_anonymization_outputs.py \
    --plan "$PLAN" --manifest "$FINAL_MANIFEST" --expected "$FULL_COUNT" \
    --finite-check-limit 100 --wall-seconds "$ANON_WALL_SECONDS" \
    --output "$RUN_ROOT/final.validation.json"
echo "anonymization_completed=$(timestamp) wall_seconds=$ANON_WALL_SECONDS"

CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_EXE" -m mmsv.cli extract-embeddings \
    --checkpoint "$CHECKPOINT" --manifest "$FINAL_MANIFEST" \
    --output "$ANONYMIZED_EMBEDDINGS"

evaluation_args=(
    --trials-dir "$TRIALS_DIR"
    --original-embeddings "$ORIGINAL_EMBEDDINGS"
    --anonymized-embeddings "$ANONYMIZED_EMBEDDINGS"
    --checkpoint "$CHECKPOINT"
    --output-root "$OUTPUT_ROOT"
)
[[ -f "$CONTROL_SUMMARY" ]] && evaluation_args+=(--control-summary "$CONTROL_SUMMARY")
"$PYTHON_EXE" scripts/evaluate_saar_session_baseline.py "${evaluation_args[@]}"

trap - INT TERM EXIT
echo "pipeline_completed=$(timestamp)"
echo "gate_result=$OUTPUT_ROOT/metrics/gate_1.json"
echo "important=This pipeline deliberately does not start SAAR training before Gate 1 is reviewed."
