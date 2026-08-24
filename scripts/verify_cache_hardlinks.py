from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def verify_hardlinks(source: Path, destination: Path, expected: int) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    source_files = {path.relative_to(source): path for path in source.glob("*/*.flac")}
    missing: list[str] = []
    different_physical_file: list[str] = []
    logical_bytes = 0
    verified = 0
    for relative, source_path in source_files.items():
        destination_path = destination / relative
        if not destination_path.is_file():
            if len(missing) < 10:
                missing.append(str(relative))
            continue
        source_stat = os.stat(source_path)
        destination_stat = os.stat(destination_path)
        logical_bytes += source_stat.st_size
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ):
            if len(different_physical_file) < 10:
                different_physical_file.append(str(relative))
        else:
            verified += 1

    valid = len(source_files) == expected and verified == expected
    return {
        "valid": valid,
        "source": str(source),
        "destination": str(destination),
        "expected": expected,
        "source_files": len(source_files),
        "verified_hardlinks": verified,
        "logical_bytes": logical_bytes,
        "missing_examples": missing,
        "different_physical_file_examples": different_physical_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that every selected-cache file is hardlinked into the full cache"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    result = verify_hardlinks(Path(args.source), Path(args.destination), args.expected)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
