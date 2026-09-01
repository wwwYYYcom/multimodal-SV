#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
MONITOR_SECONDS="${MONITOR_SECONDS:-60}"
MAX_WORKER_ATTEMPTS="${MAX_WORKER_ATTEMPTS:-20}"
MAX_TRAINING_ATTEMPTS="${MAX_TRAINING_ATTEMPTS:-10}"
DRY_RUN="${DRY_RUN:-0}"
RESERVE_GIB="${RESERVE_GIB:-12}"
FULL_COUNT=572951
PROJECTED_OUTPUT_BYTES=35828004345
RESERVE_BYTES=$((RESERVE_GIB * 1024 * 1024 * 1024))

cd "$PROJECT_ROOT"
PLAN="artifacts/anonymization/train_all_utterances_plan.csv"
FINAL_MANIFEST="artifacts/metadata/fisher_anonymized_train_corrected_manifest.csv"
STREAMVOICE_ROOT="third_party/StreamVoiceAnon"
SHARED_RUN_DIR="results/runs/anonymization_train"
SEMI_RUN_DIR="results/runs/audio_semi_corrected"
LAZY_CHECKPOINT="results/runs/audio_corrected_p1/last.pt"
SEMI_CONFIG="configs/semi_local_corrected.yaml"
SPLITS="artifacts/metadata/speaker_splits.csv"
TRIALS="artifacts/trials/evaluation.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S_%3N)"
RUN_DIR="$SHARED_RUN_DIR/multigpu_${STAMP}"
mkdir -p "$RUN_DIR" "$SEMI_RUN_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS_CSV"
if (( ${#GPU_ARRAY[@]} == 0 || WORKERS_PER_GPU <= 0 )); then
    echo "GPU_IDS and WORKERS_PER_GPU must define at least one worker" >&2
    exit 2
fi
TOTAL_WORKERS=$(( ${#GPU_ARRAY[@]} * WORKERS_PER_GPU ))

for required in "$PLAN" "${PLAN%.csv}.audit.json" "$LAZY_CHECKPOINT" "$SEMI_CONFIG" "$SPLITS" "$TRIALS"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 2
    fi
done
"$PYTHON_EXE" -c 'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); assert a["source_utterances"] == int(sys.argv[2]) and not a["one_per_call_side"]' "${PLAN%.csv}.audit.json" "$FULL_COUNT"

EXISTING_OUTPUT_BYTES="$(find artifacts/anonymized/train -type f -name '*.flac' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s+0}')"
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
if (( cursor != FULL_COUNT )); then
    echo "internal slice error: planned=$cursor expected=$FULL_COUNT" >&2
    exit 2
fi

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
        --plan "$PLAN" \
        --output-manifest "$manifest" \
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
echo "existing_output_bytes=$EXISTING_OUTPUT_BYTES projected_output_bytes=$PROJECTED_OUTPUT_BYTES startup_free_bytes=$FREE_BYTES"
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
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        exit_code=0
        wait "$pid" || exit_code=$?
        audit="${MANIFESTS[$worker]%.csv}.audit.json"
        if [[ -f "$audit" ]] && "$PYTHON_EXE" -c 'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if a["processed"] == int(sys.argv[2]) else 1)' "$audit" "${LIMITS[$worker]}"; then
            COMPLETED[$worker]=1
            incomplete=$((incomplete - 1))
            echo "worker_completed=$(timestamp) worker=$((worker + 1)) gpu=${GPUS[$worker]} attempt=${ATTEMPTS[$worker]}"
        else
            echo "worker_failed=$(timestamp) worker=$((worker + 1)) gpu=${GPUS[$worker]} attempt=${ATTEMPTS[$worker]} exit_code=$exit_code"
            if (( ATTEMPTS[$worker] >= MAX_WORKER_ATTEMPTS )); then
                echo "worker $((worker + 1)) exhausted $MAX_WORKER_ATTEMPTS attempts" >&2
                exit 1
            fi
            sleep 10
            start_worker "$worker"
        fi
    done
    (( incomplete == 0 )) && break
    now="$(date +%s)"
    if (( now >= next_report )); then
        progress_summary=""
        for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
            progress="${MANIFESTS[$worker]%.csv}.progress.jsonl"
            done_count=0
            if [[ -s "$progress" ]]; then
                last_index="$(tail -n 1 "$progress" | sed -nE 's/.*"index": ([0-9]+).*/\1/p')"
                [[ -n "$last_index" ]] && done_count=$((last_index - STARTS[$worker]))
            elif (( COMPLETED[$worker] == 1 )); then
                done_count="${LIMITS[$worker]}"
            fi
            progress_summary+=" w$((worker + 1))=${done_count}/${LIMITS[$worker]}@gpu${GPUS[$worker]}"
        done
        echo "anonymization_progress=$(timestamp)$progress_summary"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits || true
        next_report=$((now + MONITOR_SECONDS))
    fi
    sleep 5
done

ANON_WALL_SECONDS=$(( $(date +%s) - ANON_STARTED_EPOCH ))
merge_args=()
for manifest in "${MANIFESTS[@]}"; do merge_args+=("$manifest"); done
"$PYTHON_EXE" scripts/merge_anonymization_manifests.py --plan "$PLAN" --manifests "${merge_args[@]}" --output "$FINAL_MANIFEST"
"$PYTHON_EXE" scripts/validate_anonymization_outputs.py \
    --plan "$PLAN" --manifest "$FINAL_MANIFEST" --expected "$FULL_COUNT" \
    --finite-check-limit 100 --wall-seconds "$ANON_WALL_SECONDS" \
    --output "$SHARED_RUN_DIR/final.validation.json"
echo "anonymization_completed=$(timestamp) wall_seconds=$ANON_WALL_SECONDS"

SEMI_CHECKPOINT="$SEMI_RUN_DIR/last.pt"
training_complete=0
for ((attempt=1; attempt<=MAX_TRAINING_ATTEMPTS; attempt++)); do
    train_args=(-u -m mmsv.cli train-audio --config "$SEMI_CONFIG" --manifest "$FINAL_MANIFEST" --splits "$SPLITS" --output-dir "$SEMI_RUN_DIR")
    if [[ -f "$SEMI_CHECKPOINT" ]]; then
        train_args+=(--resume "$SEMI_CHECKPOINT")
        mode=resume
    else
        train_args+=(--init-from "$LAZY_CHECKPOINT")
        mode=init_from_corrected_lazy_reset_optimizer
    fi
    echo "training_started=$(timestamp) attempt=$attempt gpu=${GPU_ARRAY[0]} mode=$mode"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_EXE" "${train_args[@]}" \
        >"$SEMI_RUN_DIR/process.attempt${attempt}.stdout.log" \
        2>"$SEMI_RUN_DIR/process.attempt${attempt}.stderr.log"
    exit_code=$?
    set -e
    if [[ -f "$SEMI_CHECKPOINT" ]] && "$PYTHON_EXE" scripts/validate_training_checkpoint.py --checkpoint "$SEMI_CHECKPOINT" --expected-last-epoch 14; then
        training_complete=1
        break
    fi
    echo "training_failed=$(timestamp) attempt=$attempt exit_code=$exit_code"
    sleep 30
done
if (( training_complete != 1 )); then
    echo "semi-informed training exhausted $MAX_TRAINING_ATTEMPTS attempts" >&2
    exit 1
fi
echo "training_completed=$(timestamp)"

ORIGINAL_EMBEDDINGS="artifacts/embeddings/original_evaluation_semi_corrected.npz"
ANONYMIZED_EMBEDDINGS="artifacts/embeddings/anonymized_evaluation_semi_corrected.npz"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_EXE" -m mmsv.cli extract-embeddings \
    --checkpoint "$SEMI_CHECKPOINT" --manifest artifacts/metadata/fisher_manifest.csv \
    --trials "$TRIALS" --output "$ORIGINAL_EMBEDDINGS"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[1]:-${GPU_ARRAY[0]}}" "$PYTHON_EXE" -m mmsv.cli extract-embeddings \
    --checkpoint "$SEMI_CHECKPOINT" --manifest artifacts/metadata/fisher_anonymized_evaluation_manifest.csv \
    --trials "$TRIALS" --output "$ANONYMIZED_EMBEDDINGS"

for condition in O-A A-A; do
    if [[ "$condition" == O-A ]]; then result_dir="results/o_a_semi_corrected"; else result_dir="results/a_a_semi_corrected"; fi
    mkdir -p "$result_dir"
    for n in 1 5 10 15; do
        "$PYTHON_EXE" -m mmsv.cli score-mean --trials "$TRIALS" \
            --original-embeddings "$ORIGINAL_EMBEDDINGS" --anonymized-embeddings "$ANONYMIZED_EMBEDDINGS" \
            --condition "$condition" --n "$n" --output "$result_dir/mean_n${n}.csv"
    done
done

trap - INT TERM EXIT
echo "pipeline_completed=$(timestamp)"
echo "next_action=append semi-informed metrics and hashes to EXPERIMENT_RESULTS.md"
