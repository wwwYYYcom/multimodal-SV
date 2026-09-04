#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PROJECT_ROOT")}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$OUTPUT_DIR/mmsv_saar_gate1_results_${STAMP}.tar"

cd "$PROJECT_ROOT"
required=(
    artifacts/saar/session_baseline/evaluation_summary.json
    artifacts/saar/session_baseline/metrics
    artifacts/saar/session_baseline/figures
    artifacts/saar/session_baseline/embeddings/anonymized_evaluation_corrected.npz
    artifacts/saar/session_baseline/manifests/session_baseline_anonymized_manifest.csv
    artifacts/saar/session_baseline/manifests/session_baseline_anonymized_manifest.audit.json
    results/runs/saar_session_baseline/final.validation.json
    results/runs/saar_session_baseline/server_supervisor.log
)
for path in "${required[@]}"; do
    [[ -e "$path" ]] || { echo "missing Gate 1 result: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
tar -cf "$ARCHIVE" "${required[@]}"
sha256sum "$ARCHIVE" >"$ARCHIVE.sha256"
echo "archive=$ARCHIVE"
echo "sha256_manifest=$ARCHIVE.sha256"
echo "bytes=$(stat -c '%s' "$ARCHIVE")"
echo "audio_files_are_intentionally_excluded=true"
