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
    validation_min_calls: int = 2,
    trial_max_n: int = 15,
) -> dict[str, object]:
    calls: dict[str, set[str]] = defaultdict(set)
    utterances: dict[str, int] = defaultdict(int)
    utterances_by_call: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in iter_manifest(manifest_path):
        speaker_id = row["speaker_id"]
        calls[speaker_id].add(row["call_id"])
        utterances[speaker_id] += 1
        utterances_by_call[speaker_id][row["call_id"]] += 1

    requested = {
        "train": train_count,
        "validation": validation_count,
        "evaluation": evaluation_count,
    }
    if len(calls) < sum(requested.values()):
        if require_exact_counts:
            raise ValueError(
                f"严格论文划分需要 {sum(requested.values())} 位说话人，"
                f"manifest 只有 {len(calls)}。当前固定数据范围应使用兼容配置。"
            )
        actual = _scaled_counts(requested, len(calls))
    else:
        actual = requested

    rng = random.Random(seed)

    def trial_capable(speaker: str, min_calls: int) -> bool:
        counts = sorted(utterances_by_call[speaker].values(), reverse=True)
        if len(counts) < min_calls:
            return False
        left = 0
        right = 0
        for count in counts:
            if left <= right:
                left += count
            else:
                right += count
        return min(left, right) >= trial_max_n

    candidate_sets = {
        "evaluation": {
            speaker for speaker in calls if trial_capable(speaker, evaluation_min_calls)
        },
        "validation": {
            speaker for speaker in calls if trial_capable(speaker, validation_min_calls)
        },
    }
    selected: dict[str, set[str]] = {}
    available = set(calls)
    # 先分配 calls 要求更高的集合；要求相同时保持 evaluation -> validation 的稳定顺序。
    requirements = [
        ("evaluation", actual["evaluation"], evaluation_min_calls),
        ("validation", actual["validation"], validation_min_calls),
    ]
    requirements.sort(key=lambda item: -item[2])
    for split_name, count, min_calls in requirements:
        candidates = sorted(candidate_sets[split_name] & available)
        if len(candidates) < count:
            raise ValueError(
                f"{split_name} 需要 {count} 位可构造 call-disjoint max_n={trial_max_n} "
                f"trial 的说话人（至少 {min_calls} calls），实际可分配 {len(candidates)}"
            )
        rng.shuffle(candidates)
        selected[split_name] = set(candidates[:count])
        available -= selected[split_name]

    evaluation = selected["evaluation"]
    validation = selected["validation"]
    remaining = sorted(available)
    rng.shuffle(remaining)
    train = set(remaining[: actual["train"]])
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
        "validation_min_calls": validation_min_calls,
        "trial_max_n": trial_max_n,
        "evaluation_candidates": len(candidate_sets["evaluation"]),
        "validation_candidates": len(candidate_sets["validation"]),
        "disjoint": len(assigned) == len(train) + len(validation) + len(evaluation),
    }
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit
