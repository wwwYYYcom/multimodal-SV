from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_variant_plan(
    source_rows: list[dict[str, str]],
    fieldnames: list[str],
    path: Path,
    audio_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_rows:
            copied = dict(row)
            copied["output_audio_path"] = str(
                (audio_root / row["speaker_id"] / f"{row['utt_id']}.flac").resolve()
            )
            writer.writerow(copied)


def _progress_metrics(path: Path, warmup: int) -> dict[str, float | int]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    measured = records[min(warmup, len(records)) :]
    elapsed = sum(float(record["elapsed_seconds"]) for record in measured)
    audio = sum(float(record["seconds"]) for record in measured)
    return {
        "items": len(records),
        "warmup_excluded": len(records) - len(measured),
        "measured_items": len(measured),
        "measured_elapsed_seconds": elapsed,
        "measured_audio_seconds": audio,
        "steady_rtf": elapsed / audio if audio else float("inf"),
        "steady_items_per_second": len(measured) / elapsed if elapsed else 0.0,
    }


def _audio_validation(plan_path: Path) -> dict[str, object]:
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_rates: set[int] = set()
    unreadable: list[str] = []
    nonfinite: list[str] = []
    for row in rows:
        output = Path(row["output_audio_path"])
        try:
            info = sf.info(output)
            sample_rates.add(int(info.samplerate))
            audio, _ = sf.read(output, dtype="float32")
            if not np.isfinite(audio).all():
                nonfinite.append(str(output))
        except Exception:
            unreadable.append(str(output))
    return {
        "expected": len(rows),
        "readable": len(rows) - len(unreadable),
        "sample_rates": sorted(sample_rates),
        "unreadable": unreadable,
        "nonfinite": nonfinite,
        "valid": not unreadable and not nonfinite and sample_rates == {16000},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark StreamVoiceAnon torch.compile flags")
    parser.add_argument("--plan", default="artifacts/anonymization/evaluation_plan.csv")
    parser.add_argument("--streamvoice-root", default="third_party/StreamVoiceAnon")
    parser.add_argument("--start-index", type=int, default=303)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--run-root", default="results/runs/anonymization_compile_benchmark")
    parser.add_argument("--audio-root", default="artifacts/anonymized/compile_benchmark")
    parser.add_argument("--compile-cache-root", default=r"D:\mmsv_tc")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "baseline",
            "compiled_all",
            "compiled_ar",
            "compiled_encoder",
            "compiled_decoder",
            "compiled_helpers",
            "fp16",
        ],
        default=["baseline", "compiled_all"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_plan = (project_root / args.plan).resolve()
    with source_plan.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        all_rows = list(reader)
    selected = all_rows[args.start_index : args.start_index + args.count]
    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} rows but only selected {len(selected)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (project_root / args.run_root / stamp).resolve()
    audio_root = (project_root / args.audio_root / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, object] = {
        "started": datetime.now().astimezone().isoformat(),
        "source_plan": str(source_plan),
        "start_index": args.start_index,
        "count": args.count,
        "warmup": args.warmup,
        "run_dir": str(run_dir),
        "audio_root": str(audio_root),
        "variants": {},
    }

    available_variants = {
        "baseline": [],
        "compiled_all": ["--compile-ar", "--compile-encoder", "--compile-decoder"],
        "compiled_ar": ["--compile-ar"],
        "compiled_encoder": ["--compile-encoder"],
        "compiled_decoder": ["--compile-decoder"],
        "compiled_helpers": ["--compile-encoder", "--compile-decoder"],
        "fp16": ["--fp16"],
    }
    variants = {name: available_variants[name] for name in args.variants}
    for name, flags in variants.items():
        variant_plan = run_dir / f"{name}.plan.csv"
        manifest = run_dir / f"{name}.manifest.csv"
        _write_variant_plan(selected, fieldnames, variant_plan, audio_root / name)
        command = [
            sys.executable,
            "-u",
            "-m",
            "mmsv.cli",
            "anonymize-streamvoice",
            "--plan",
            str(variant_plan),
            "--output-manifest",
            str(manifest),
            "--streamvoice-root",
            str((project_root / args.streamvoice_root).resolve()),
            "--delay",
            "2",
            "--alpha",
            "1.0",
            *flags,
        ]
        print(f"variant_started={name} time={datetime.now().astimezone().isoformat()}", flush=True)
        started = time.perf_counter()
        with (run_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / f"{name}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            environment = os.environ.copy()
            if flags:
                cache_root = Path(args.compile_cache_root).resolve()
                temp_root = cache_root / "tmp"
                inductor_root = cache_root / "inductor"
                triton_root = cache_root / "triton"
                for directory in (temp_root, inductor_root, triton_root):
                    directory.mkdir(parents=True, exist_ok=True)
                environment.update({
                    "TEMP": str(temp_root),
                    "TMP": str(temp_root),
                    "TORCHINDUCTOR_CACHE_DIR": str(inductor_root),
                    "TRITON_CACHE_DIR": str(triton_root),
                })
                variant_result_cache = {
                    "temp": str(temp_root),
                    "torchinductor": str(inductor_root),
                    "triton": str(triton_root),
                }
            else:
                variant_result_cache = None
            completed = subprocess.run(
                command,
                cwd=project_root,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
        wall_seconds = time.perf_counter() - started
        variant_result: dict[str, object] = {
            "command": command,
            "return_code": completed.returncode,
            "wall_seconds": wall_seconds,
            "plan": str(variant_plan),
            "manifest": str(manifest),
            "compile_cache": variant_result_cache,
        }
        progress = manifest.with_suffix(".progress.jsonl")
        if completed.returncode == 0 and progress.exists():
            variant_result["performance"] = _progress_metrics(progress, args.warmup)
            variant_result["validation"] = _audio_validation(variant_plan)
        results["variants"][name] = variant_result
        (run_dir / "benchmark.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"variant_completed={name} return_code={completed.returncode} wall_seconds={wall_seconds:.3f}",
            flush=True,
        )
        if completed.returncode != 0:
            break

    baseline = results["variants"].get("baseline", {})
    compiled = results["variants"].get("compiled_all", {})
    if baseline.get("performance") and compiled.get("performance"):
        baseline_rtf = baseline["performance"]["steady_rtf"]
        compiled_rtf = compiled["performance"]["steady_rtf"]
        speedup = baseline_rtf / compiled_rtf
        results["comparison"] = {
            "speedup": speedup,
            "rtf_reduction_fraction": 1.0 - (compiled_rtf / baseline_rtf),
            "switch_threshold_fraction": 0.15,
            "recommend_compiled": speedup >= (1.0 / 0.85)
            and bool(compiled.get("validation", {}).get("valid")),
        }
    results["completed"] = datetime.now().astimezone().isoformat()
    output = run_dir / "benchmark.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
