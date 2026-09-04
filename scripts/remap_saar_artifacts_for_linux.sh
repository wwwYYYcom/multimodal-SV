#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
SERVER_PROJECT_ROOT="${SERVER_PROJECT_ROOT:-$PROJECT_ROOT}"
SERVER_CORPORA_ROOT="${SERVER_CORPORA_ROOT:-/public/home/wwwyyycom123_/datasets/corpora}"
LOCAL_PROJECT_ROOT="${LOCAL_PROJECT_ROOT:-D:/deeplearning/ICASSP2027/multimodal_sv_reproduction}"
LOCAL_CORPORA_ROOT="${LOCAL_CORPORA_ROOT:-D:/deeplearning/realtimeVoiceAnon/dataset/prefor_vpc2024/Voice-Privacy-Challenge-2024-main/corpora}"
PLAN="artifacts/saar/session_baseline/manifests/session_baseline_anonymization_plan.csv"

cd "$PROJECT_ROOT"
[[ -f "$PLAN" ]] || { echo "missing SAAR plan: $PLAN" >&2; exit 2; }

backup="${PLAN}.windows-source"
if [[ ! -f "$backup" ]]; then
    cp -p "$PLAN" "$backup"
fi

"$PYTHON_EXE" scripts/remap_csv_paths.py \
    --input "$backup" \
    --output "$PLAN" \
    --mapping "$LOCAL_CORPORA_ROOT=$SERVER_CORPORA_ROOT" \
    --mapping "$LOCAL_PROJECT_ROOT=$SERVER_PROJECT_ROOT"

"$PYTHON_EXE" - "$PLAN" "$SERVER_PROJECT_ROOT" "$SERVER_CORPORA_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

plan, project_root, corpora_root = map(Path, sys.argv[1:])
rows = list(csv.DictReader(plan.open(encoding="utf-8", newline="")))
assert len(rows) == 66712, len(rows)
assert all(row["session_id"] for row in rows)
assert all(Path(row["audio_path"]).is_relative_to(corpora_root) for row in rows)
assert all(Path(row["reference_audio_path"]).is_relative_to(corpora_root) for row in rows)
assert all(Path(row["output_audio_path"]).is_relative_to(project_root) for row in rows)
print("saar_linux_path_remap_ready=true")
print(f"rows={len(rows)}")
PY
