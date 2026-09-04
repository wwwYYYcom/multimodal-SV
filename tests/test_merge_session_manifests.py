import csv
import json
import subprocess
import sys
from pathlib import Path


def test_merge_preserves_session_id(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utt_id"])
        writer.writeheader()
        writer.writerows([{"utt_id": "u2"}, {"utt_id": "u1"}])

    fields = [
        "utt_id", "speaker_id", "call_id", "channel", "session_id",
        "audio_path", "start", "end", "duration", "transcript",
    ]
    worker = tmp_path / "worker.manifest.csv"
    with worker.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for utterance_id in ["u1", "u2"]:
            writer.writerow({
                "utt_id": utterance_id,
                "speaker_id": "s1",
                "call_id": "c1",
                "channel": "0",
                "session_id": "c1:0",
                "audio_path": f"{utterance_id}.flac",
                "start": "0",
                "end": "1",
                "duration": "1",
                "transcript": "fixture",
            })
    worker.with_suffix(".audit.json").write_text(
        json.dumps({
            "processed": 2,
            "generated": 2,
            "skipped_existing": 0,
            "generated_chunked_utterances": 0,
            "generated_inference_chunks": 2,
            "max_source_chunk_seconds": 30.0,
        }),
        encoding="utf-8",
    )

    output = tmp_path / "merged.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/merge_anonymization_manifests.py",
            "--plan", str(plan),
            "--manifests", str(worker),
            "--output", str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["utt_id"] for row in rows] == ["u2", "u1"]
    assert {row["session_id"] for row in rows} == {"c1:0"}
