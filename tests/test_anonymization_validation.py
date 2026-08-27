import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.validate_anonymization_outputs import validate_outputs


def test_validate_anonymization_outputs(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    plan = tmp_path / "plan.csv"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["utt_id", "duration", "output_audio_path"]
        )
        writer.writeheader()
        writer.writerow({"utt_id": "one", "duration": "0.1", "output_audio_path": audio})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utt_id"])
        writer.writeheader()
        writer.writerow({"utt_id": "one"})

    result = validate_outputs(
        plan,
        manifest,
        expected=1,
        wall_seconds=0.2,
        full_plan_source_hours=1.0,
    )
    assert result["valid"] is True
    assert result["real_time_factor"] == 2.0
    assert result["projected_full_wall_hours"] == 2.0
