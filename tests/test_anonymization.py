import csv
import json
from pathlib import Path

import numpy as np

import mmsv.anonymization as anonymization
from mmsv.anonymization import anonymize_plan, build_anonymization_plan
from mmsv.cli import build_parser


def test_anonymization_plan_is_trial_filtered_and_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "utt_id", "speaker_id", "call_id", "channel", "audio_path",
            "start", "end", "duration", "transcript",
        ])
        writer.writeheader()
        for index in range(3):
            writer.writerow({
                "utt_id": f"u{index}", "speaker_id": "s", "call_id": "c",
                "channel": "0", "audio_path": f"source-{index}.wav", "start": "0",
                "end": "2", "duration": "2", "transcript": "hello",
            })
    pool = tmp_path / "pool.csv"
    with pool.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "utt_id", "speaker_id", "audio_path", "duration", "subset",
        ])
        writer.writeheader()
        for index in range(4):
            writer.writerow({
                "utt_id": f"r{index}", "speaker_id": f"rs{index}",
                "audio_path": f"ref-{index}.flac", "duration": "5", "subset": "train-clean-360",
            })
    trials = tmp_path / "trials.jsonl"
    trials.write_text(json.dumps({
        "enroll_utt_ids": ["u0"], "target_utt_ids": ["u2"],
    }) + "\n", encoding="utf-8")

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    result = build_anonymization_plan(manifest, pool, first, tmp_path / "audio", 1234, trials)
    build_anonymization_plan(manifest, pool, second, tmp_path / "audio", 1234, trials)
    assert result["source_utterances"] == 2
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    with first.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["utt_id"] for row in rows} == {"u0", "u2"}
    assert all(row["output_audio_path"].endswith(".flac") for row in rows)


def test_anonymization_plan_can_select_one_source_per_call_side(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "utt_id", "speaker_id", "call_id", "channel", "audio_path",
            "start", "end", "duration", "transcript",
        ])
        writer.writeheader()
        for channel in range(2):
            for index in range(3):
                writer.writerow({
                    "utt_id": f"u-{channel}-{index}", "speaker_id": f"s{channel}",
                    "call_id": "c", "channel": str(channel), "audio_path": "call.sph",
                    "start": "0", "end": "2", "duration": "2", "transcript": "hello",
                })
    pool = tmp_path / "pool.csv"
    with pool.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "utt_id", "speaker_id", "audio_path", "duration", "subset",
        ])
        writer.writeheader()
        writer.writerow({
            "utt_id": "r", "speaker_id": "rs", "audio_path": "ref.flac",
            "duration": "5", "subset": "train-clean-360",
        })
    splits = tmp_path / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker_id", "split"])
        writer.writeheader()
        writer.writerows([
            {"speaker_id": "s0", "split": "train"},
            {"speaker_id": "s1", "split": "train"},
        ])

    output = tmp_path / "plan.csv"
    result = build_anonymization_plan(
        manifest,
        pool,
        output,
        tmp_path / "audio",
        seed=1234,
        split_csv=splits,
        split_name="train",
        one_per_call_side=True,
    )
    assert result["source_utterances"] == 2
    assert result["one_per_call_side"] is True


def test_anonymization_cli_accepts_compile_and_slice_flags() -> None:
    args = build_parser().parse_args([
        "anonymize-streamvoice",
        "--plan", "plan.csv",
        "--output-manifest", "manifest.csv",
        "--streamvoice-root", "StreamVoiceAnon",
        "--start-index", "303",
        "--limit", "20",
        "--compile-ar",
        "--compile-encoder",
        "--compile-decoder",
        "--fp16",
    ])
    assert args.start_index == 303
    assert args.limit == 20
    assert args.compile_ar is True
    assert args.compile_encoder is True
    assert args.compile_decoder is True
    assert args.fp16 is True


def test_anonymize_plan_streams_requested_slice_and_writes_atomic_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeWrapper:
        device = "cpu"
        sr = 16000

        @staticmethod
        def infer(*args, **kwargs):
            return np.full(320, 0.1, dtype=np.float32)

    monkeypatch.setattr(anonymization, "_load_streamvoice_wrapper", lambda *args, **kwargs: FakeWrapper())
    monkeypatch.setattr(
        anonymization,
        "read_segment",
        lambda *args, **kwargs: np.full(320, 0.1, dtype=np.float32),
    )
    plan = tmp_path / "plan.csv"
    fieldnames = [
        "utt_id", "speaker_id", "call_id", "channel", "audio_path",
        "start", "end", "duration", "transcript", "reference_utt_id",
        "reference_speaker_id", "reference_audio_path", "reference_duration",
        "output_audio_path",
    ]
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(3):
            writer.writerow({
                "utt_id": f"u{index}", "speaker_id": "s", "call_id": "c",
                "channel": "0", "audio_path": "source.wav", "start": "0",
                "end": "0.02", "duration": "0.02", "transcript": "hello",
                "reference_utt_id": "r", "reference_speaker_id": "rs",
                "reference_audio_path": "reference.wav", "reference_duration": "5",
                "output_audio_path": str(tmp_path / f"u{index}.flac"),
            })
    manifest = tmp_path / "manifest.csv"
    result = anonymize_plan(
        plan,
        manifest,
        tmp_path,
        start_index=1,
        limit=1,
    )
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert result["processed"] == 1
    assert result["start_index"] == 1
    assert [row["utt_id"] for row in rows] == ["u1"]
    assert not manifest.with_suffix(".csv.tmp").exists()
