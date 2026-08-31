from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint anonymization worker manifests")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifests", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    worker_audits: list[dict[str, object]] = []
    fieldnames = [
        "utt_id",
        "speaker_id",
        "call_id",
        "channel",
        "audio_path",
        "start",
        "end",
        "duration",
        "transcript",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    processed = 0
    with tempfile.TemporaryDirectory(
        prefix="mmsv_manifest_merge_", dir=args.output.parent
    ) as temporary_dir:
        database_path = Path(temporary_dir) / "rows.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE rows (utt_id TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID"
            )
            try:
                for manifest in args.manifests:
                    with manifest.open("r", encoding="utf-8", newline="") as handle:
                        connection.executemany(
                            "INSERT INTO rows VALUES (?, ?)",
                            (
                                (row["utt_id"], json.dumps(row, ensure_ascii=False))
                                for row in csv.DictReader(handle)
                            ),
                        )
                    audit_path = manifest.with_suffix(".audit.json")
                    worker_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate utterance across workers: {error}") from error
            worker_row_count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
            missing: list[str] = []
            with (
                args.plan.open("r", encoding="utf-8", newline="") as plan_handle,
                temporary.open("w", encoding="utf-8", newline="") as output_handle,
            ):
                writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
                writer.writeheader()
                for plan_row in csv.DictReader(plan_handle):
                    utterance_id = plan_row["utt_id"]
                    match = connection.execute(
                        "SELECT payload FROM rows WHERE utt_id = ?", (utterance_id,)
                    ).fetchone()
                    if match is None:
                        if len(missing) < 10:
                            missing.append(utterance_id)
                        continue
                    writer.writerow(json.loads(match[0]))
                    processed += 1
            extras = worker_row_count - processed
            if missing or extras:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"worker manifests do not match plan: missing={missing}, extra_count={extras}"
                )
    temporary.replace(args.output)

    audit = {
        "plan": str(args.plan.resolve()),
        "manifest": str(args.output.resolve()),
        "worker_manifests": [str(path.resolve()) for path in args.manifests],
        "device": "cuda:dual-process",
        "processed": processed,
        "generated": sum(int(item["generated"]) for item in worker_audits),
        "skipped_existing": sum(int(item["skipped_existing"]) for item in worker_audits),
        "generated_chunked_utterances": sum(
            int(item.get("generated_chunked_utterances", 0)) for item in worker_audits
        ),
        "generated_inference_chunks": sum(
            int(item.get("generated_inference_chunks", 0)) for item in worker_audits
        ),
        "sample_rate": 16000,
        "delay": 2,
        "alpha": 1.0,
        "fp16": False,
        "max_source_chunk_seconds": worker_audits[0].get("max_source_chunk_seconds"),
        "workers": worker_audits,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
