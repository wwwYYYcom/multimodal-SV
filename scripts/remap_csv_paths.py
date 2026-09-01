from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")


def _normalise(value: str) -> str:
    return value.replace("\\", "/")


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path mapping must use OLD=NEW")
    old, new = value.split("=", 1)
    old = _normalise(old).rstrip("/")
    new = _normalise(new).rstrip("/")
    if not old or not new:
        raise argparse.ArgumentTypeError("both OLD and NEW path prefixes are required")
    return old, new


def remap_path(value: str, mappings: list[tuple[str, str]]) -> tuple[str, bool]:
    normalised = _normalise(value)
    folded = normalised.casefold()
    for old, new in sorted(mappings, key=lambda item: len(item[0]), reverse=True):
        old_folded = old.casefold()
        if folded == old_folded:
            return new, True
        prefix = old_folded + "/"
        if folded.startswith(prefix):
            return new + normalised[len(old) :], True
    return normalised, False


def remap_csv(
    input_path: Path,
    output_path: Path,
    mappings: list[tuple[str, str]],
    allow_unmapped_absolute: bool = False,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    path_values = 0
    mapped_values = 0
    unmapped_examples: list[str] = []
    with (
        input_path.open("r", encoding="utf-8", newline="") as source,
        temporary.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        path_columns = {
            name for name in reader.fieldnames if name == "audio_path" or name.endswith("_path")
        }
        if not path_columns:
            raise ValueError(f"CSV has no path columns: {input_path}")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            rows += 1
            for column in path_columns:
                value = row.get(column, "")
                if not value:
                    continue
                path_values += 1
                rewritten, mapped = remap_path(value, mappings)
                if mapped:
                    mapped_values += 1
                elif WINDOWS_ABSOLUTE.match(value) or os.path.isabs(value):
                    if len(unmapped_examples) < 10:
                        unmapped_examples.append(f"row={rows} column={column} value={value}")
                row[column] = rewritten
            writer.writerow(row)
    if unmapped_examples and not allow_unmapped_absolute:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            "absolute paths were not covered by a mapping: " + "; ".join(unmapped_examples)
        )
    temporary.replace(output_path)
    result = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": rows,
        "path_values": path_values,
        "mapped_values": mapped_values,
        "unmapped_absolute_values": len(unmapped_examples),
        "unmapped_examples": unmapped_examples,
        "mappings": [{"old": old, "new": new} for old, new in mappings],
    }
    output_path.with_suffix(output_path.suffix + ".remap.audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap absolute CSV paths for another host")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", required=True, action="append", type=parse_mapping)
    parser.add_argument("--allow-unmapped-absolute", action="store_true")
    args = parser.parse_args()
    result = remap_csv(
        args.input,
        args.output,
        args.mapping,
        allow_unmapped_absolute=args.allow_unmapped_absolute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
