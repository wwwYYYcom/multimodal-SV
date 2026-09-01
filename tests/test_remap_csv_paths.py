import csv
import json
from pathlib import Path

import pytest

from scripts.remap_csv_paths import remap_csv, remap_path


def test_remap_path_prefers_longest_case_insensitive_prefix() -> None:
    result, mapped = remap_path(
        r"D:\Data\LibriSpeech\train-clean-360\1\a.flac",
        [
            ("D:/Data", "/srv/data"),
            ("D:/Data/LibriSpeech/train-clean-360", "/srv/librispeech"),
        ],
    )
    assert mapped is True
    assert result == "/srv/librispeech/1/a.flac"


def test_remap_csv_streams_paths_and_writes_audit(tmp_path: Path) -> None:
    source = tmp_path / "plan.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utt_id", "audio_path", "reference_audio_path", "output_audio_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "utt_id": "u1",
                "audio_path": r"D:\Fisher\call.sph",
                "reference_audio_path": r"D:\Libri\ref.flac",
                "output_audio_path": r"D:\Project\artifacts\out.flac",
            }
        )
    output = tmp_path / "server.csv"
    result = remap_csv(
        source,
        output,
        [
            ("D:/Fisher", "/data/fisher"),
            ("D:/Libri", "/data/libri"),
            ("D:/Project", "/work/project"),
        ],
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["audio_path"] == "/data/fisher/call.sph"
    assert row["reference_audio_path"] == "/data/libri/ref.flac"
    assert row["output_audio_path"] == "/work/project/artifacts/out.flac"
    assert result["rows"] == 1
    assert result["mapped_values"] == 3
    audit = json.loads(output.with_suffix(".csv.remap.audit.json").read_text(encoding="utf-8"))
    assert audit["unmapped_absolute_values"] == 0


def test_remap_csv_rejects_unmapped_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "manifest.csv"
    source.write_text("utt_id,audio_path\nu1,C:\\\\unknown\\\\a.wav\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not covered"):
        remap_csv(source, tmp_path / "output.csv", [("D:/known", "/data")])


def test_remap_csv_accepts_target_prefix_self_mapping(tmp_path: Path) -> None:
    source = tmp_path / "server.csv"
    source.write_text(
        "utt_id,audio_path,output_audio_path\n"
        "u1,/data/fisher/call.sph,/work/project/artifacts/u1.flac\n",
        encoding="utf-8",
    )
    result = remap_csv(
        source,
        source,
        [
            ("D:/Corpora", "/data"),
            ("D:/Project", "/work/project"),
            ("/data", "/data"),
            ("/work/project", "/work/project"),
        ],
    )
    with source.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["audio_path"] == "/data/fisher/call.sph"
    assert row["output_audio_path"] == "/work/project/artifacts/u1.flac"
    assert result["mapped_values"] == 2
    assert result["unmapped_absolute_values"] == 0
