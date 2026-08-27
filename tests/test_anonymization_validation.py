import csv
import json
import subprocess
import sys
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


def test_merge_anonymization_manifests_restores_plan_order(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utt_id"])
        writer.writeheader()
        writer.writerows([{"utt_id": "one"}, {"utt_id": "two"}])

    fields = [
        "utt_id", "speaker_id", "call_id", "channel", "audio_path",
        "start", "end", "duration", "transcript",
    ]
    manifests = []
    for name, utterance_id in [("worker1.csv", "two"), ("worker2.csv", "one")]:
        manifest = tmp_path / name
        manifests.append(manifest)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({field: utterance_id if field == "utt_id" else "x" for field in fields})
        manifest.with_suffix(".audit.json").write_text(
            json.dumps({"generated": 1, "skipped_existing": 0}), encoding="utf-8"
        )

    output = tmp_path / "merged.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "merge_anonymization_manifests.py"
    subprocess.run(
        [
            sys.executable, str(script), "--plan", str(plan), "--manifests",
            *(str(path) for path in manifests), "--output", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        assert [row["utt_id"] for row in csv.DictReader(handle)] == ["one", "two"]
    audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
    assert audit["processed"] == 2
    assert audit["generated"] == 2
