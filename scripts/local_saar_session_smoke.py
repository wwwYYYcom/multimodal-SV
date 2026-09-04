from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def _prepare_plan(source_plan: Path, run_dir: Path, count: int) -> tuple[Path, list[dict[str, str]]]:
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    fieldnames: list[str] | None = None
    with source_plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            sessions[row["session_id"]].append(row)
    candidates = [rows for rows in sessions.values() if len(rows) >= count]
    if not candidates or fieldnames is None:
        raise ValueError(f"no session contains {count} plan utterances")
    selected = min(
        candidates,
        key=lambda rows: sum(float(row["duration"]) for row in rows[:count]),
    )[:count]
    reference_ids = {row["reference_utt_id"] for row in selected}
    reference_paths = {row["reference_audio_path"] for row in selected}
    if len(reference_ids) != 1 or len(reference_paths) != 1:
        raise ValueError("selected session does not use one fixed pseudo reference")
    audio_root = run_dir / "audio"
    for row in selected:
        row["output_audio_path"] = str(
            (audio_root / row["speaker_id"] / f"{row['utt_id']}.flac").resolve()
        )
    output_plan = run_dir / "session_smoke.plan.csv"
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    with output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return output_plan, selected


def _validate(
    rows: list[dict[str, str]],
    manifest: Path,
    max_relative_duration_error: float,
) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if [row["utt_id"] for row in manifest_rows] != [row["utt_id"] for row in rows]:
        raise ValueError("smoke manifest order/IDs do not match plan")
    outputs = []
    for row in rows:
        path = Path(row["output_audio_path"])
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        duration = len(audio) / sample_rate
        expected = float(row["duration"])
        relative_error = abs(duration - expected) / expected
        if sample_rate != 16000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError(f"invalid smoke audio: {path}")
        if relative_error >= max_relative_duration_error:
            raise ValueError(f"duration mismatch for {row['utt_id']}: {relative_error}")
        outputs.append({
            "utt_id": row["utt_id"],
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_rate": sample_rate,
            "frames": len(audio),
            "duration": duration,
            "expected_duration": expected,
            "relative_duration_error": relative_error,
            "finite": True,
        })
    return {
        "valid": True,
        "session_id": rows[0]["session_id"],
        "speaker_id": rows[0]["speaker_id"],
        "reference_utt_id": rows[0]["reference_utt_id"],
        "reference_speaker_id": rows[0]["reference_speaker_id"],
        "utterances": len(rows),
        "max_relative_duration_error_allowed": max_relative_duration_error,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real local session-fixed StreamVoiceAnon smoke")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--streamvoice-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--max-relative-duration-error",
        type=float,
        default=0.05,
        help="StreamVoiceAnon output-duration tolerance; 5%% covers the audited full evaluation",
    )
    args = parser.parse_args()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = args.output_root / stamp
    plan, rows = _prepare_plan(args.plan, run_dir, args.count)
    manifest = run_dir / "session_smoke.manifest.csv"
    stdout = run_dir / "anonymize.stdout.log"
    stderr = run_dir / "anonymize.stderr.log"
    command = [
        sys.executable, "-u", "-m", "mmsv.cli", "anonymize-streamvoice",
        "--plan", str(plan),
        "--output-manifest", str(manifest),
        "--streamvoice-root", str(args.streamvoice_root),
        "--delay", "2", "--alpha", "1.0",
        "--max-source-chunk-seconds", "30.0",
    ]
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"StreamVoiceAnon smoke failed; see {stderr}")
    result = _validate(rows, manifest, args.max_relative_duration_error)
    result.update({
        "run_dir": str(run_dir.resolve()),
        "plan": str(plan.resolve()),
        "manifest": str(manifest.resolve()),
        "stdout_log": str(stdout.resolve()),
        "stderr_log": str(stderr.resolve()),
        "command": command,
    })
    output = run_dir / "validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
