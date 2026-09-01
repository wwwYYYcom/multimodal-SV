#!/usr/bin/env bash
set -Eeuo pipefail

MMSV_HOME="${MMSV_HOME:-/public/home/wwwyyycom123_/multimodal_sv_reproduction}"
MMSV_VENV="${MMSV_VENV:-/public/home/wwwyyycom123_/venvs/mmsv}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python}"

if [[ ! -f "$MMSV_HOME/pyproject.toml" ]]; then
    echo "project checkout not found: $MMSV_HOME" >&2
    exit 2
fi

"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
mkdir -p "$(dirname "$MMSV_VENV")"
if [[ ! -x "$MMSV_VENV/bin/python" ]]; then
    if ! "$BOOTSTRAP_PYTHON" -m venv "$MMSV_VENV"; then
        echo "venv creation failed; on Ubuntu install python3-venv and rerun" >&2
        exit 2
    fi
fi

PYTHON_EXE="$MMSV_VENV/bin/python"
"$PYTHON_EXE" -m pip install --upgrade --retries 10 --timeout 120 pip setuptools wheel
"$PYTHON_EXE" -m pip install --retries 10 --timeout 120 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_EXE" -m pip install --retries 10 --timeout 120 -e "$MMSV_HOME[audio,test]"
"$PYTHON_EXE" -m pip install --retries 10 --timeout 120 \
    accelerate==1.1.1 \
    einx==0.4.3 \
    einops==0.8.0 \
    hydra-core==1.3.2 \
    huggingface-hub==0.36.2 \
    librosa==0.11.0 \
    matplotlib==3.10.6 \
    munch==4.0.0 \
    numpy==1.26.4 \
    pandas==2.3.3 \
    pyarrow==23.0.0 \
    pydub==0.25.1 \
    pyloudnorm==0.2.0 \
    python-dotenv==1.2.2 \
    safetensors==0.6.2 \
    scikit-learn==1.7.2 \
    scipy==1.13.1 \
    sentencepiece==0.2.1 \
    soundfile==0.13.1 \
    speechbrain==0.5.16 \
    tqdm==4.67.1 \
    transformers==4.56.2 \
    vector-quantize-pytorch==1.14.24

cd "$MMSV_HOME"
"$PYTHON_EXE" -m pytest -q
(
    cd third_party/StreamVoiceAnon
    "$PYTHON_EXE" -c 'from evaluations.infer_arvc import InferenceWrapper; print("streamvoice_import=true")'
)

AUDIT_DIR="$MMSV_HOME/results/runs/server_setup"
mkdir -p "$AUDIT_DIR"
"$PYTHON_EXE" - "$AUDIT_DIR/environment.json" "$MMSV_HOME" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy
import scipy
import soundfile
import torch
import torchaudio
import transformers

output = Path(sys.argv[1])
payload = {
    "completed_at": datetime.now().astimezone().isoformat(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": sys.version,
    "venv": sys.prefix,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchaudio": torchaudio.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "soundfile": soundfile.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpus": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ],
    "git_head": subprocess.check_output(
        ["git", "-C", sys.argv[2], "rev-parse", "HEAD"],
        text=True,
    ).strip(),
}
if payload["torch"] != "2.9.1+cu128" or payload["torchaudio"] != "2.9.1+cu128":
    raise SystemExit(f"unexpected torch stack: {payload['torch']} / {payload['torchaudio']}")
if not payload["cuda_available"] or payload["cuda_device_count"] != 2:
    raise SystemExit(f"expected two CUDA GPUs: {payload}")
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "server_environment_ready=true"
echo "python_exe=$PYTHON_EXE"
echo "audit=$AUDIT_DIR/environment.json"
