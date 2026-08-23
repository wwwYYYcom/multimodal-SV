from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def build_librispeech_pool(
    roots: Iterable[str | Path], output_csv: str | Path, min_duration: float = 4.0
) -> dict[str, object]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("构建 LibriSpeech pool 需要 soundfile") from exc

    rows: list[dict[str, object]] = []
    root_names: list[str] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        root_names.append(root.name)
        for path in sorted(root.rglob("*.flac")):
            info = sf.info(path)
            duration = float(info.frames / info.samplerate)
            if duration <= min_duration:  # 论文原文是 strictly longer than 4 seconds。
                continue
            parts = path.stem.split("-")
            rows.append({
                "utt_id": path.stem,
                "speaker_id": parts[0],
                "audio_path": str(path),
                "duration": round(duration, 3),
                "subset": root.name,
            })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["utt_id", "speaker_id", "audio_path", "duration", "subset"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "pool": str(output_path.resolve()),
        "subsets": root_names,
        "utterances": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "min_duration_strictly_greater_than": min_duration,
        "paper_requires_both_subsets": {"train-clean-360", "train-other-500"}.issubset(root_names),
    }
