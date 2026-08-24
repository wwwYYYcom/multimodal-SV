import os
from pathlib import Path

from scripts.verify_cache_hardlinks import verify_hardlinks


def test_verify_cache_hardlinks_accepts_shared_physical_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "ab").mkdir(parents=True)
    (destination / "ab").mkdir(parents=True)
    original = source / "ab" / "one.flac"
    linked = destination / "ab" / "one.flac"
    original.write_bytes(b"audio")
    os.link(original, linked)
    result = verify_hardlinks(source, destination, expected=1)
    assert result["valid"] is True
    assert result["verified_hardlinks"] == 1


def test_verify_cache_hardlinks_rejects_copies(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "ab").mkdir(parents=True)
    (destination / "ab").mkdir(parents=True)
    (source / "ab" / "one.flac").write_bytes(b"audio")
    (destination / "ab" / "one.flac").write_bytes(b"audio")
    result = verify_hardlinks(source, destination, expected=1)
    assert result["valid"] is False
    assert result["verified_hardlinks"] == 0
