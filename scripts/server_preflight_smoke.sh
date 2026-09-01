#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1}"
HF_HOME="${HF_HOME:-/public/home/wwwyyycom123_/.cache/huggingface}"
PLAN="${PLAN:-$PROJECT_ROOT/artifacts/anonymization/train_all_utterances_plan.csv}"
STREAMVOICE_ROOT="${STREAMVOICE_ROOT:-$PROJECT_ROOT/third_party/StreamVoiceAnon}"
STAMP="$(date +%Y%m%d_%H%M%S_%3N)"
RUN_DIR="${SMOKE_RUN_DIR:-$PROJECT_ROOT/results/runs/server_preflight_smoke/$STAMP}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS_CSV"
if (( ${#GPU_ARRAY[@]} < 2 )); then
    echo "GPU_IDS must contain at least two GPU IDs" >&2
    exit 2
fi
GPU_NORMAL="${GPU_ARRAY[0]}"
GPU_LONG="${GPU_ARRAY[1]}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"

for required in \
    "$PLAN" \
    "$STREAMVOICE_ROOT/pretrained_checkpoints/asr_s2s_bsq_8192_causal_down_whisper.pth" \
    "$STREAMVOICE_ROOT/pretrained_checkpoints/campplus_cn_common.bin" \
    "$STREAMVOICE_ROOT/pretrained_checkpoints/dual_ar_delay_0_8.pth" \
    "$STREAMVOICE_ROOT/pretrained_checkpoints/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    "$STREAMVOICE_ROOT/pretrained_checkpoints/spark_speaker_encoder.pth" \
    "$PROJECT_ROOT/results/runs/audio_corrected_p1/last.pt"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required smoke input: $required" >&2
        exit 2
    fi
done

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "preflight_started=$(date --iso-8601=seconds)" | tee "$RUN_DIR/preflight.log"
echo "project_root=$PROJECT_ROOT" | tee -a "$RUN_DIR/preflight.log"
echo "run_dir=$RUN_DIR" | tee -a "$RUN_DIR/preflight.log"
echo "gpu_normal=$GPU_NORMAL gpu_long=$GPU_LONG" | tee -a "$RUN_DIR/preflight.log"

"$PYTHON_EXE" - "$GPU_NORMAL" "$GPU_LONG" <<'PY' | tee "$RUN_DIR/environment.json"
import json
import sys

import torch
from transformers import AutoConfig

gpu_ids = [int(sys.argv[1]), int(sys.argv[2])]
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() > max(gpu_ids), (
    torch.cuda.device_count(), gpu_ids
)
config = AutoConfig.from_pretrained("microsoft/wavlm-large", local_files_only=True)
assert config.model_type == "wavlm"
assert config.hidden_size == 1024
assert config.num_hidden_layers == 24
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in gpu_ids
    ],
    "wavlm": {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "offline": True,
    },
}, ensure_ascii=False, indent=2))
PY

sha256sum -c <<EOF | tee "$RUN_DIR/weights.sha256.log"
dd02fc319d66216159693f6523ebbc4262afd43f41630de70599cf77f99b159e  $STREAMVOICE_ROOT/pretrained_checkpoints/asr_s2s_bsq_8192_causal_down_whisper.pth
3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8  $STREAMVOICE_ROOT/pretrained_checkpoints/campplus_cn_common.bin
df703a1a710c807ad0651dd1bbe45556bf5f3a47f1a79929ec3e6e8fecc56583  $STREAMVOICE_ROOT/pretrained_checkpoints/dual_ar_delay_0_8.pth
01b81dbf753224a156c3fe139b88bf0b9a0f54b11bee864f95e66511c3ccd754  $STREAMVOICE_ROOT/pretrained_checkpoints/firefly-gan-vq-fsq-8x1024-21hz-generator.pth
84adb871ada3c41ac54b8c4897b88c2ed80962937e283437f9a392980ffd3483  $STREAMVOICE_ROOT/pretrained_checkpoints/spark_speaker_encoder.pth
0c69749dbb51929054e3e57990b04d2e737cefd96902f1d0100e80b402313508  $PROJECT_ROOT/results/runs/audio_corrected_p1/last.pt
fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d  $HF_HOME/hub/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c/pytorch_model.bin
EOF

"$PYTHON_EXE" - "$PLAN" "$RUN_DIR" <<'PY' | tee "$RUN_DIR/selection.json"
import csv
import json
import re
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
windows_absolute = re.compile(r"^[A-Za-z]:[/\\]")
normal = None
long_sample = None
long_index = None
with plan_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"plan has no header: {plan_path}")
    for index, row in enumerate(reader):
        if normal is None:
            normal = row.copy()
        if row["utt_id"] == "fe_03_00170_B_0060":
            long_sample = row.copy()
            long_index = index

if normal is None or long_sample is None:
    raise RuntimeError("normal or long smoke row was not found")
if long_index != 13930:
    raise RuntimeError(f"unexpected long sample index: {long_index}")

selected = {"normal": normal, "long": long_sample}
summary = {}
for name, row in selected.items():
    for key in ("audio_path", "reference_audio_path"):
        value = row[key]
        if windows_absolute.match(value) or not Path(value).is_absolute():
            raise RuntimeError(f"plan is not remapped: {name} {key}={value}")
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    output = run_dir / name / row["speaker_id"] / f"{row['utt_id']}.flac"
    row["output_audio_path"] = str(output)
    smoke_plan = run_dir / f"{name}.plan.csv"
    with smoke_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    summary[name] = {
        "utt_id": row["utt_id"],
        "source_duration": float(row["duration"]),
        "source_path": row["audio_path"],
        "reference_path": row["reference_audio_path"],
        "output_path": row["output_audio_path"],
        "plan": str(smoke_plan),
        "original_plan_index": 0 if name == "normal" else long_index,
    }
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits | tee "$RUN_DIR/nvidia.before.csv"

CUDA_VISIBLE_DEVICES="$GPU_NORMAL" "$PYTHON_EXE" -u -m mmsv.cli anonymize-streamvoice \
    --plan "$RUN_DIR/normal.plan.csv" \
    --output-manifest "$RUN_DIR/normal.manifest.csv" \
    --streamvoice-root "$STREAMVOICE_ROOT" \
    --delay 2 --alpha 1.0 --max-source-chunk-seconds 30.0 \
    >"$RUN_DIR/normal.stdout.log" 2>"$RUN_DIR/normal.stderr.log" &
normal_pid=$!

CUDA_VISIBLE_DEVICES="$GPU_LONG" "$PYTHON_EXE" -u -m mmsv.cli anonymize-streamvoice \
    --plan "$RUN_DIR/long.plan.csv" \
    --output-manifest "$RUN_DIR/long.manifest.csv" \
    --streamvoice-root "$STREAMVOICE_ROOT" \
    --delay 2 --alpha 1.0 --max-source-chunk-seconds 30.0 \
    >"$RUN_DIR/long.stdout.log" 2>"$RUN_DIR/long.stderr.log" &
long_pid=$!

echo "normal_pid=$normal_pid long_pid=$long_pid" | tee -a "$RUN_DIR/preflight.log"
set +e
wait "$normal_pid"
normal_status=$?
wait "$long_pid"
long_status=$?
set -e
echo "normal_exit=$normal_status long_exit=$long_status" | tee -a "$RUN_DIR/preflight.log"
if (( normal_status != 0 || long_status != 0 )); then
    tail -n 80 "$RUN_DIR/normal.stderr.log" >&2 || true
    tail -n 80 "$RUN_DIR/long.stderr.log" >&2 || true
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits | tee "$RUN_DIR/nvidia.after.csv"

"$PYTHON_EXE" - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/validation.json"
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

run_dir = Path(sys.argv[1]).resolve()
result = {}
for name in ("normal", "long"):
    plan_path = run_dir / f"{name}.plan.csv"
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    output = Path(row["output_audio_path"])
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"missing output: {output}")
    info = sf.info(output)
    waveform, sample_rate = sf.read(output, dtype="float32", always_2d=True)
    duration = len(waveform) / sample_rate
    expected_duration = float(row["duration"])
    relative_error = abs(duration - expected_duration) / expected_duration
    audit_path = run_dir / f"{name}.manifest.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert sample_rate == 16000
    assert info.channels == 1
    assert np.isfinite(waveform).all()
    assert relative_error < 0.02, (name, duration, expected_duration)
    assert audit["processed"] == 1
    assert audit["generated"] == 1
    assert audit["device"] == "cuda"
    if name == "normal":
        assert audit["generated_chunked_utterances"] == 0
        assert audit["generated_inference_chunks"] == 1
    else:
        assert audit["generated_chunked_utterances"] == 1
        assert audit["generated_inference_chunks"] == 2
    result[name] = {
        "utt_id": row["utt_id"],
        "output_path": str(output),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sample_rate": sample_rate,
        "channels": info.channels,
        "frames": len(waveform),
        "duration": duration,
        "expected_duration": expected_duration,
        "relative_duration_error": relative_error,
        "finite": True,
        "generated_chunked_utterances": audit["generated_chunked_utterances"],
        "generated_inference_chunks": audit["generated_inference_chunks"],
        "audit_path": str(audit_path),
        "stdout_log": str(run_dir / f"{name}.stdout.log"),
        "stderr_log": str(run_dir / f"{name}.stderr.log"),
    }
print(json.dumps({
    "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    "preflight_smoke_ready": True,
    "results": result,
}, ensure_ascii=False, indent=2))
PY

echo "preflight_completed=$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/preflight.log"
echo "server_preflight_smoke_ready=true"
echo "run_dir=$RUN_DIR"
