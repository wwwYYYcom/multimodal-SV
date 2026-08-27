from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def validate_outputs(
    plan_path: Path,
    manifest_path: Path,
    expected: int,
    sample_rate: int = 16000,
    finite_check_limit: int = 100,
    wall_seconds: float | None = None,
    full_plan_source_hours: float | None = None,
) -> dict[str, object]:
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))[:expected]
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    manifest_for = {row["utt_id"]: row for row in manifest_rows}
    missing: list[str] = []
    unreadable: list[str] = []
    wrong_format: list[str] = []
    nonfinite: list[str] = []
    duration_relative_errors: list[float] = []
    output_bytes = 0
    source_seconds = 0.0
    output_seconds = 0.0

    for index, row in enumerate(plan_rows):
        utterance_id = row["utt_id"]
        source_duration = float(row["duration"])
        source_seconds += source_duration
        destination = Path(row["output_audio_path"])
        manifest_row = manifest_for.get(utterance_id)
        if manifest_row is None or not destination.is_file() or destination.stat().st_size <= 0:
            if len(missing) < 10:
                missing.append(utterance_id)
            continue
        try:
            info = sf.info(destination)
            output_bytes += destination.stat().st_size
            output_seconds += float(info.duration)
            if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
                if len(wrong_format) < 10:
                    wrong_format.append(utterance_id)
            if source_duration > 0:
                duration_relative_errors.append(
                    abs(float(info.duration) - source_duration) / source_duration
                )
            if index < finite_check_limit:
                audio, read_rate = sf.read(destination, dtype="float32", always_2d=False)
                if read_rate != sample_rate or not np.isfinite(audio).all() or audio.size == 0:
                    if len(nonfinite) < 10:
                        nonfinite.append(utterance_id)
        except Exception as error:  # pragma: no cover - depends on decoder failure mode
            if len(unreadable) < 10:
                unreadable.append(f"{utterance_id}: {error}")

    ordered_plan_ids = [row["utt_id"] for row in plan_rows]
    ordered_manifest_ids = [row["utt_id"] for row in manifest_rows]
    id_order_matches = ordered_manifest_ids == ordered_plan_ids
    valid = (
        len(plan_rows) == expected
        and len(manifest_rows) == expected
        and len(manifest_for) == expected
        and id_order_matches
        and not missing
        and not unreadable
        and not wrong_format
        and not nonfinite
    )
    real_time_factor = None
    projected_full_wall_hours = None
    if wall_seconds is not None and source_seconds > 0:
        real_time_factor = wall_seconds / source_seconds
        if full_plan_source_hours is not None:
            projected_full_wall_hours = real_time_factor * full_plan_source_hours
    projected_full_bytes = None
    if full_plan_source_hours is not None and source_seconds > 0:
        projected_full_bytes = round(
            output_bytes / source_seconds * full_plan_source_hours * 3600.0
        )

    return {
        "valid": valid,
        "plan": str(plan_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "expected": expected,
        "plan_rows": len(plan_rows),
        "manifest_rows": len(manifest_rows),
        "unique_manifest_ids": len(manifest_for),
        "id_order_matches": id_order_matches,
        "source_seconds": source_seconds,
        "output_seconds": output_seconds,
        "output_bytes": output_bytes,
        "duration_relative_error_p50": _percentile(duration_relative_errors, 50),
        "duration_relative_error_p95": _percentile(duration_relative_errors, 95),
        "duration_relative_error_max": max(duration_relative_errors, default=None),
        "finite_files_checked": min(expected, finite_check_limit),
        "missing_examples": missing,
        "unreadable_examples": unreadable,
        "wrong_format_examples": wrong_format,
        "nonfinite_examples": nonfinite,
        "wall_seconds": wall_seconds,
        "real_time_factor": real_time_factor,
        "projected_full_wall_hours": projected_full_wall_hours,
        "projected_full_bytes": projected_full_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate StreamVoiceAnon output files")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--finite-check-limit", type=int, default=100)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--full-plan-source-hours", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_outputs(
        args.plan,
        args.manifest,
        args.expected,
        args.sample_rate,
        args.finite_check_limit,
        args.wall_seconds,
        args.full_plan_source_hours,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
