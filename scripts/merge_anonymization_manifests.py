from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint anonymization worker manifests")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifests", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.plan.open("r", encoding="utf-8", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    rows_for_id: dict[str, dict[str, str]] = {}
    worker_audits: list[dict[str, object]] = []
    for manifest in args.manifests:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                utterance_id = row["utt_id"]
                if utterance_id in rows_for_id:
                    raise ValueError(f"duplicate utterance across workers: {utterance_id}")
                rows_for_id[utterance_id] = row
        audit_path = manifest.with_suffix(".audit.json")
        worker_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))

    plan_ids = [row["utt_id"] for row in plan_rows]
    missing = [utterance_id for utterance_id in plan_ids if utterance_id not in rows_for_id]
    extras = sorted(set(rows_for_id) - set(plan_ids))
    if missing or extras:
        raise ValueError(
            f"worker manifests do not match plan: missing={missing[:10]}, extras={extras[:10]}"
        )

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
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_id[utterance_id] for utterance_id in plan_ids)
    temporary.replace(args.output)

    audit = {
        "plan": str(args.plan.resolve()),
        "manifest": str(args.output.resolve()),
        "worker_manifests": [str(path.resolve()) for path in args.manifests],
        "device": "cuda:dual-process",
        "processed": len(plan_rows),
        "generated": sum(int(item["generated"]) for item in worker_audits),
        "skipped_existing": sum(int(item["skipped_existing"]) for item in worker_audits),
        "sample_rate": 16000,
        "delay": 2,
        "alpha": 1.0,
        "fp16": False,
        "workers": worker_audits,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
