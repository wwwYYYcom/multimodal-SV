#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_EXE="${PYTHON_EXE:-/public/home/wwwyyycom123_/venvs/mmsv/bin/python}"
if pgrep -f '[m]msv.cli (anonymize-streamvoice|train-audio|extract-embeddings)' >/dev/null; then
    echo 'Existing GPU task detected; finish it before starting robustness run' >&2
    exit 2
fi
mkdir -p results/runs/saar_robustness
exec 9>results/runs/saar_robustness/run.lock
flock -n 9 || { echo 'Robustness runner already active'; exit 2; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="artifacts/saar/robustness_${STAMP}"
LOG="results/runs/saar_robustness/${STAMP}.log"
printf '%s\n' "$OUT" >results/runs/saar_robustness/output.latest
printf '%s\n' "$LOG" >results/runs/saar_robustness/log.latest
"$PYTHON_EXE" -u scripts/saar_robustness.py --mode run --output "$OUT" \
    --gpus "${GPU_IDS:-0,1,2,3}" --replicates "${REPLICATES:-1000}" 2>&1 | tee "$LOG"
ARCHIVE="${OUT##*/}_results.tar"
tar -cf "$(dirname "$PWD")/$ARCHIVE" "$OUT" "$LOG" EXPERIMENT_RESULTS.md
sha256sum "$(dirname "$PWD")/$ARCHIVE" | tee "$(dirname "$PWD")/$ARCHIVE.sha256"
echo "archive=$(dirname "$PWD")/$ARCHIVE"
