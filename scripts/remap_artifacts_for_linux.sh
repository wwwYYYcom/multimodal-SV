#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
CORPORA_ROOT="${CORPORA_ROOT:?set CORPORA_ROOT to the Linux directory corresponding to the local corpora directory}"
WINDOWS_CORPORA_ROOT="${WINDOWS_CORPORA_ROOT:-D:/deeplearning/realtimeVoiceAnon/dataset/prefor_vpc2024/Voice-Privacy-Challenge-2024-main/corpora}"
WINDOWS_PROJECT_ROOT="${WINDOWS_PROJECT_ROOT:-D:/deeplearning/ICASSP2027/multimodal_sv_reproduction}"

cd "$PROJECT_ROOT"
files=(
    artifacts/anonymization/train_all_utterances_plan.csv
    artifacts/metadata/fisher_manifest.csv
    artifacts/metadata/librispeech_target_pool.csv
    artifacts/metadata/fisher_anonymized_evaluation_manifest.csv
)

for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "missing required CSV: $file" >&2
        exit 2
    fi
    "$PYTHON_EXE" scripts/remap_csv_paths.py \
        --input "$file" --output "$file" \
        --mapping "$WINDOWS_CORPORA_ROOT=$CORPORA_ROOT" \
        --mapping "$WINDOWS_PROJECT_ROOT=$PROJECT_ROOT"
done

echo "remap_completed=true project_root=$PROJECT_ROOT corpora_root=$CORPORA_ROOT"
