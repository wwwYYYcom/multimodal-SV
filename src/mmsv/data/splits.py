from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .fisher import iter_manifest


def _scaled_counts(requested: dict[str, int], available: int) -> dict[str, int]:
    total = sum(requested.values())
    raw = {name: available * value / total for name, value in requested.items()}
    result = {name: int(value) for name, value in raw.items()}
    remaining = available - sum(result.values())
    order = sorted(raw, key=lambda name: raw[name] - result[name], reverse=True)
    for name in order[:remaining]:
        result[name] += 1
    return result


def split_speakers(
    manifest_path: str | Path,
    output_csv: str | Path,
    train_count: int,
    validation_count: int,
    evaluation_count: int,
    seed: int,
    require_exact_counts: bool,
    evaluation_min_calls: int = 2,
) -> dict[str, object]:
    calls: dict[str, set[str]] = defaultdict(set)
    utterances: dict[str, int] = defaultdict(int)
    for row in iter_manifest(manifest_path):
        speaker_id = row["speaker_id"]
        calls[speaker_id].add(row["call_id"])
        utterances[speaker_id] += 1

    requested = {
        "train": train_count,
        "validation": validation_count,
        "evaluation": evaluation_count,
    }
    if len(calls) < sum(requested.values()):
        if require_exact_counts:
            raise ValueError(
                f"严格论文划分需要 {sum(requested.values())} 位说话人，"
                f"manifest 只有 {len(calls)}。请补齐 Fisher Part 2，或使用兼容配置。"
            )
        actual = _scaled_counts(requested, len(calls))
    else:
        actual = requested

    eval_candidates = sorted(
        speaker for speaker, speaker_calls in calls.items()
        if len(speaker_calls) >= evaluation_min_calls
    )
    if len(eval_candidates) < actual["evaluation"]:
        raise ValueError(
            f"evaluation 需要 {actual['evaluation']} 位至少 {evaluation_min_calls} calls 的说话人，"
            f"实际只有 {len(eval_candidates)}"
        )

    rng = random.Random(seed)
    rng.shuffle(eval_candidates)
    evaluation = set(eval_candidates[: actual["evaluation"]])
    remaining = sorted(set(calls) - evaluation)
    rng.shuffle(remaining)
    validation = set(remaining[: actual["validation"]])
    train = set(remaining[actual["validation"] : actual["validation"] + actual["train"]])
    assigned = train | validation | evaluation
    if assigned != set(calls):
        # 精确配置在数据比论文更多时，把多余 speaker 保留在 train，避免静默丢数据。
        train.update(set(calls) - assigned)
        actual["train"] = len(train)

    split_for = {speaker: "train" for speaker in train}
    split_for.update({speaker: "validation" for speaker in validation})
    split_for.update({speaker: "evaluation" for speaker in evaluation})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["speaker_id", "split", "n_calls", "n_utterances"]
        )
        writer.writeheader()
        for speaker in sorted(split_for):
            writer.writerow(
                {
                    "speaker_id": speaker,
                    "split": split_for[speaker],
                    "n_calls": len(calls[speaker]),
                    "n_utterances": utterances[speaker],
                }
            )

    audit = {
        "speaker_split": str(output_path.resolve()),
        "seed": seed,
        "seed_source": "reproduction choice; not reported by the paper",
        "requested_counts": requested,
        "actual_counts": {name: sum(value == name for value in split_for.values()) for name in requested},
        "require_exact_counts": require_exact_counts,
        "evaluation_min_calls": evaluation_min_calls,
        "evaluation_candidates": len(eval_candidates),
        "disjoint": len(assigned) == len(train) + len(validation) + len(evaluation),
    }
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit

