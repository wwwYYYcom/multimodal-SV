from __future__ import annotations

import csv
import itertools
import json
import os
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .audio import read_segment
from .data.fisher import iter_manifest


def trial_utterance_ids(path: str | Path) -> set[str]:
    utterance_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            trial = json.loads(line)
            utterance_ids.update(map(str, trial["enroll_utt_ids"]))
            utterance_ids.update(map(str, trial["target_utt_ids"]))
    return utterance_ids


def _stable_reference_index(seed: int, utterance_id: str, pool_size: int) -> int:
    return random.Random(f"mmsv:{seed}:{utterance_id}").randrange(pool_size)


def build_anonymization_plan(
    manifest_path: str | Path,
    reference_pool_csv: str | Path,
    output_csv: str | Path,
    audio_output_root: str | Path,
    seed: int,
    trial_jsonl: str | Path | None = None,
    limit: int | None = None,
    split_csv: str | Path | None = None,
    split_name: str | None = None,
    one_per_call_side: bool = False,
) -> dict[str, object]:
    with Path(reference_pool_csv).open("r", encoding="utf-8", newline="") as handle:
        references = sorted(csv.DictReader(handle), key=lambda row: row["utt_id"])
    if not references:
        raise ValueError("LibriSpeech reference pool 为空")

    if trial_jsonl is not None and split_csv is not None:
        raise ValueError("trial filter 与 split filter 不能同时使用")
    if (split_csv is None) != (split_name is None):
        raise ValueError("split_csv 与 split_name 必须同时提供")
    requested = trial_utterance_ids(trial_jsonl) if trial_jsonl is not None else None
    split_for: dict[str, str] | None = None
    if split_csv is not None:
        with Path(split_csv).open("r", encoding="utf-8", newline="") as handle:
            split_for = {row["speaker_id"]: row["split"] for row in csv.DictReader(handle)}
    source_rows = [
        row for row in iter_manifest(manifest_path)
        if requested is None or row["utt_id"] in requested
    ]
    if split_for is not None:
        source_rows = [
            row for row in source_rows if split_for.get(row["speaker_id"]) == split_name
        ]
    if requested is not None:
        found = {row["utt_id"] for row in source_rows}
        missing = requested - found
        if missing:
            raise KeyError(f"trial 中有 {len(missing)} 条 utterance 不在 manifest，例如 {min(missing)}")
    if one_per_call_side:
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in source_rows:
            groups.setdefault((row["audio_path"], row["channel"]), []).append(row)
        selected_ids = {
            rows[_stable_reference_index(seed, f"source:{audio_path}:{channel}", len(rows))]["utt_id"]
            for (audio_path, channel), rows in groups.items()
        }
        source_rows = [row for row in source_rows if row["utt_id"] in selected_ids]
    if limit is not None:
        source_rows = source_rows[:limit]
    if not source_rows:
        raise ValueError("匿名化计划没有 source utterance")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root = Path(audio_output_root).resolve()
    fieldnames = [
        "utt_id", "speaker_id", "call_id", "channel", "audio_path", "start", "end",
        "duration", "transcript", "reference_utt_id", "reference_speaker_id",
        "reference_audio_path", "reference_duration", "output_audio_path",
    ]
    reference_ids: set[str] = set()
    total_seconds = 0.0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_rows:
            reference = references[_stable_reference_index(seed, row["utt_id"], len(references))]
            reference_ids.add(reference["utt_id"])
            total_seconds += float(row["duration"])
            destination = audio_root / row["speaker_id"] / f"{row['utt_id']}.flac"
            writer.writerow({
                **{name: row[name] for name in fieldnames[:9]},
                "reference_utt_id": reference["utt_id"],
                "reference_speaker_id": reference["speaker_id"],
                "reference_audio_path": reference["audio_path"],
                "reference_duration": reference["duration"],
                "output_audio_path": str(destination),
            })

    audit = {
        "plan": str(output_path.resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "reference_pool": str(Path(reference_pool_csv).resolve()),
        "trial_filter": None if trial_jsonl is None else str(Path(trial_jsonl).resolve()),
        "split_filter": None if split_csv is None else str(Path(split_csv).resolve()),
        "split_name": split_name,
        "one_per_call_side": one_per_call_side,
        "seed": seed,
        "mapping": "per_utterance_deterministic_random",
        "source_utterances": len(source_rows),
        "source_hours": total_seconds / 3600.0,
        "reference_pool_utterances": len(references),
        "unique_references_selected": len(reference_ids),
        "audio_output_root": str(audio_root),
    }
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_streamvoice_wrapper(
    streamvoice_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    compile_ar: bool = False,
    compile_encoder: bool = False,
    compile_decoder: bool = False,
    fp16: bool = False,
) -> Any:
    root_text = str(streamvoice_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    with _working_directory(streamvoice_root):
        from evaluations.infer_arvc import InferenceWrapper

        if compile_ar or compile_encoder or compile_decoder:
            # The upstream module enables coordinate-descent autotuning globally.
            # On this Windows environment its static CUDA launcher overflows a C long.
            # Disabling it changes kernel selection only, not model computation.
            import torch

            torch._inductor.config.coordinate_descent_tuning = False

        wrapper = InferenceWrapper(
            str(config_path),
            str(checkpoint_path),
            compile_ar=compile_ar,
            compile_encoder=compile_encoder,
            compile_decoder=compile_decoder,
            fp16=fp16,
        )
        if compile_ar or compile_encoder or compile_decoder:
            # ARVCWrapper.compile_ar_decode_fn() turns this back on while it creates
            # the lazy compiled callable, so apply the Windows-safe value again.
            torch._inductor.config.coordinate_descent_tuning = False
    if wrapper.device.type == "cpu":
        # 上游固定使用 FP16 KV cache；CPU autocast 会关闭并产生 FP32 source。
        decoder_model = wrapper.model.decoder.model
        decoder_model.max_seq_len = -1
        decoder_model.max_batch_size = -1
        wrapper.model.setup_caches(max_batch_size=1, max_seq_len=2048, dtype=__import__("torch").float32)
    return wrapper


def anonymize_plan(
    plan_csv: str | Path,
    output_manifest: str | Path,
    streamvoice_root: str | Path,
    config_path: str | Path = "configs/config_firefly_arvcasr_8192_delay0_8.yaml",
    checkpoint_path: str | Path = "pretrained_checkpoints/dual_ar_delay_0_8.pth",
    sample_rate: int = 16000,
    delay: int = 2,
    alpha: float = 1.0,
    sph2pipe: str | None = None,
    limit: int | None = None,
    start_index: int = 0,
    compile_ar: bool = False,
    compile_encoder: bool = False,
    compile_decoder: bool = False,
    fp16: bool = False,
) -> dict[str, object]:
    import soundfile as sf
    from scipy.signal import resample_poly

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    root = Path(streamvoice_root).resolve()
    config = Path(config_path)
    checkpoint = Path(checkpoint_path)
    if not config.is_absolute():
        config = root / config
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    wrapper = _load_streamvoice_wrapper(
        root,
        config,
        checkpoint,
        compile_ar=compile_ar,
        compile_encoder=compile_encoder,
        compile_decoder=compile_decoder,
        fp16=fp16,
    )

    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = manifest_path.with_suffix(".progress.jsonl")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    fieldnames = [
        "utt_id", "speaker_id", "call_id", "channel", "audio_path",
        "start", "end", "duration", "transcript",
    ]
    stop_index = None if limit is None else start_index + max(0, limit)
    processed = 0
    generated = 0
    skipped = 0
    with (
        Path(plan_csv).open("r", encoding="utf-8", newline="") as plan_handle,
        temporary_manifest.open("w", encoding="utf-8", newline="") as manifest_handle,
        progress_path.open("a", encoding="utf-8", buffering=1) as progress,
    ):
        rows = itertools.islice(csv.DictReader(plan_handle), start_index, stop_index)
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=start_index + 1):
            item_started = time.perf_counter()
            destination = Path(row["output_audio_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size > 0:
                skipped += 1
            else:
                source_row = {
                    name: row[name]
                    for name in [
                        "utt_id", "speaker_id", "call_id", "channel", "audio_path",
                        "start", "end", "duration", "transcript",
                    ]
                }
                waveform = read_segment(source_row, sample_rate, sph2pipe)
                with tempfile.TemporaryDirectory(prefix="mmsv_streamvoice_") as temporary:
                    temp_root = Path(temporary)
                    source_wav = temp_root / "source.wav"
                    sf.write(source_wav, np.asarray(waveform, dtype=np.float32), sample_rate)
                    with _working_directory(root):
                        generated_audio = wrapper.infer(
                            str(source_wav),
                            row["reference_audio_path"],
                            delay=delay,
                            alpha=alpha,
                            save_result=False,
                        )
                    audio = np.asarray(generated_audio, dtype=np.float32)
                    generated_rate = int(wrapper.sr)
                    if generated_rate != sample_rate:
                        divisor = int(np.gcd(generated_rate, sample_rate))
                        audio = resample_poly(
                            audio,
                            sample_rate // divisor,
                            generated_rate // divisor,
                        ).astype(np.float32)
                    temporary_output = destination.with_suffix(destination.suffix + ".tmp")
                    sf.write(temporary_output, audio, sample_rate, format="FLAC")
                    temporary_output.replace(destination)
                generated += 1

            info = sf.info(destination)
            writer.writerow({
                "utt_id": row["utt_id"],
                "speaker_id": row["speaker_id"],
                "call_id": row["call_id"],
                "channel": "0",
                "audio_path": str(destination.resolve()),
                "start": "0",
                "end": f"{info.duration:.8f}",
                "duration": f"{info.duration:.8f}",
                "transcript": row["transcript"],
            })
            progress.write(json.dumps({
                "index": index,
                "utt_id": row["utt_id"],
                "output": str(destination.resolve()),
                "seconds": info.duration,
                "elapsed_seconds": time.perf_counter() - item_started,
            }, ensure_ascii=False) + "\n")
            processed += 1
    if processed == 0:
        temporary_manifest.unlink(missing_ok=True)
        raise ValueError("匿名化计划为空")
    temporary_manifest.replace(manifest_path)
    result = {
        "plan": str(Path(plan_csv).resolve()),
        "manifest": str(manifest_path.resolve()),
        "progress": str(progress_path.resolve()),
        "device": str(wrapper.device),
        "processed": processed,
        "generated": generated,
        "skipped_existing": skipped,
        "sample_rate": sample_rate,
        "delay": delay,
        "alpha": alpha,
        "start_index": start_index,
        "compile_ar": compile_ar,
        "compile_encoder": compile_encoder,
        "compile_decoder": compile_decoder,
        "fp16": fp16,
    }
    manifest_path.with_suffix(".audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
