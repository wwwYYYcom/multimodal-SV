from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .fisher import iter_manifest


def nested_sample(pool: Sequence[str], n_values: Sequence[int], rng: random.Random) -> dict[int, list[str]]:
    """一次采 max(N)，再取前缀，保证 U5 ⊂ U10 ⊂ U15。"""
    ordered_n = sorted(set(int(value) for value in n_values))
    if not ordered_n or ordered_n[0] <= 0:
        raise ValueError("n_values 必须为正整数")
    max_n = ordered_n[-1]
    if len(pool) < max_n:
        raise ValueError(f"pool 只有 {len(pool)} 条，少于 max_n={max_n}")
    selected = rng.sample(list(pool), max_n)
    return {n: selected[:n] for n in ordered_n}


def _speaker_pools(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """按 call 分池；贪心平衡 utterance 数，保证两侧没有 call 泄漏。"""
    by_call: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_call[row["call_id"]].append(row["utt_id"])
    if len(by_call) < 2:
        return [], []
    left: list[str] = []
    right: list[str] = []
    for _, utt_ids in sorted(by_call.items(), key=lambda item: (-len(item[1]), item[0])):
        (left if len(left) <= len(right) else right).extend(sorted(utt_ids))
    return left, right


def read_splits(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["speaker_id"]: row["split"] for row in csv.DictReader(handle)}


def build_trials(
    manifest_path: str | Path,
    split_path: str | Path,
    output_jsonl: str | Path,
    split_name: str,
    n_values: Sequence[int],
    target_per_speaker: int,
    nontarget_per_speaker: int,
    seed: int,
) -> dict[str, object]:
    split_for = read_splits(split_path)
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in iter_manifest(manifest_path):
        if split_for.get(row["speaker_id"]) == split_name:
            by_speaker[row["speaker_id"]].append(row)

    max_n = max(n_values)
    pools = {speaker: _speaker_pools(rows) for speaker, rows in by_speaker.items()}
    eligible = sorted(
        speaker for speaker, (enroll, target) in pools.items()
        if len(enroll) >= max_n and len(target) >= max_n
    )
    if len(eligible) < 2:
        raise ValueError(f"{split_name} 中可构造双侧 max_n={max_n} trial 的 speaker 少于 2")

    rng = random.Random(seed)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"target": 0, "nontarget": 0}
    with output_path.open("w", encoding="utf-8") as handle:
        for speaker in eligible:
            enroll_pool, target_pool = pools[speaker]
            for index in range(target_per_speaker):
                enroll = nested_sample(enroll_pool, n_values, rng)[max_n]
                target = nested_sample(target_pool, n_values, rng)[max_n]
                handle.write(json.dumps({
                    "trial_id": f"{split_name}-{speaker}-tar-{index:03d}",
                    "label": 1,
                    "enroll_speaker": speaker,
                    "target_speaker": speaker,
                    "enroll_utt_ids": enroll,
                    "target_utt_ids": target,
                }, ensure_ascii=False) + "\n")
                counts["target"] += 1
            impostors = [candidate for candidate in eligible if candidate != speaker]
            for index in range(nontarget_per_speaker):
                target_speaker = rng.choice(impostors)
                enroll = nested_sample(enroll_pool, n_values, rng)[max_n]
                target = nested_sample(pools[target_speaker][1], n_values, rng)[max_n]
                handle.write(json.dumps({
                    "trial_id": f"{split_name}-{speaker}-non-{index:03d}",
                    "label": 0,
                    "enroll_speaker": speaker,
                    "target_speaker": target_speaker,
                    "enroll_utt_ids": enroll,
                    "target_utt_ids": target,
                }, ensure_ascii=False) + "\n")
                counts["nontarget"] += 1

    audit = {
        "trial_file": str(output_path.resolve()),
        "split": split_name,
        "seed": seed,
        "n_values": sorted(set(n_values)),
        "nested_sampling": True,
        "call_disjoint_pools": True,
        "eligible_speakers": len(eligible),
        "ineligible_speakers": len(by_speaker) - len(eligible),
        **counts,
    }
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def validate_trials(
    path: str | Path,
    n_values: Sequence[int],
    manifest_path: str | Path | None = None,
) -> dict[str, int]:
    max_n = max(n_values)
    counts = {"target": 0, "nontarget": 0}
    utterance_meta: dict[str, tuple[str, str]] | None = None
    if manifest_path is not None:
        utterance_meta = {
            row["utt_id"]: (row["speaker_id"], row["call_id"])
            for row in iter_manifest(manifest_path)
        }
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            enroll = row["enroll_utt_ids"]
            target = row["target_utt_ids"]
            if len(enroll) != max_n or len(target) != max_n:
                raise ValueError(f"trial 第 {line_number} 行不是 max_n={max_n}")
            if len(set(enroll)) != max_n or len(set(target)) != max_n:
                raise ValueError(f"trial 第 {line_number} 行含重复 utterance")
            if utterance_meta is not None:
                missing = [utt_id for utt_id in enroll + target if utt_id not in utterance_meta]
                if missing:
                    raise KeyError(f"trial 第 {line_number} 行引用未知 utterance: {missing[0]}")
                enroll_speakers = {utterance_meta[utt_id][0] for utt_id in enroll}
                target_speakers = {utterance_meta[utt_id][0] for utt_id in target}
                if enroll_speakers != {row["enroll_speaker"]}:
                    raise ValueError(f"trial 第 {line_number} 行 enrollment speaker 标签不一致")
                if target_speakers != {row["target_speaker"]}:
                    raise ValueError(f"trial 第 {line_number} 行 target speaker 标签不一致")
                if int(row["label"]) == 1:
                    enroll_calls = {utterance_meta[utt_id][1] for utt_id in enroll}
                    target_calls = {utterance_meta[utt_id][1] for utt_id in target}
                    if enroll_calls & target_calls:
                        raise ValueError(f"trial 第 {line_number} 行 target trial 存在 call 泄漏")
            for smaller, larger in zip(sorted(n_values), sorted(n_values)[1:]):
                if not set(enroll[:smaller]).issubset(enroll[:larger]):
                    raise ValueError("enrollment nested sampling 失效")
                if not set(target[:smaller]).issubset(target[:larger]):
                    raise ValueError("target nested sampling 失效")
            counts["target" if int(row["label"]) == 1 else "nontarget"] += 1
    return counts
