from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels 与 scores 必须是一维同形数组")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("EER 同时需要 target 与 non-target trials")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_accepts = np.cumsum(sorted_labels == 1)
    false_accepts = np.cumsum(sorted_labels == 0)
    fnr = np.concatenate(([1.0], 1.0 - true_accepts / positives))
    fpr = np.concatenate(([0.0], false_accepts / negatives))
    thresholds = np.concatenate(([np.inf], sorted_scores))
    difference = fpr - fnr
    crossing = np.where(difference >= 0)[0]
    index = int(crossing[0]) if crossing.size else len(difference) - 1
    if index == 0:
        return float((fpr[0] + fnr[0]) / 2), float(thresholds[0])
    previous = index - 1
    denominator = difference[index] - difference[previous]
    weight = 0.0 if denominator == 0 else -difference[previous] / denominator
    eer = fpr[previous] + weight * (fpr[index] - fpr[previous])
    threshold = thresholds[index]
    if np.isfinite(thresholds[previous]) and np.isfinite(thresholds[index]):
        threshold = thresholds[previous] + weight * (thresholds[index] - thresholds[previous])
    return float(eer), float(threshold)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    ids = archive["utt_ids"].astype(str)
    embeddings = archive["embeddings"].astype(np.float32)
    if len(ids) != len(embeddings):
        raise ValueError(f"embedding 文件长度不一致: {path}")
    return dict(zip(ids.tolist(), embeddings))


def _aggregate(ids: list[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = [utt_id for utt_id in ids if utt_id not in embeddings]
    if missing:
        raise KeyError(f"缺少 {len(missing)} 个 embedding，例如 {missing[0]}")
    vector = np.mean([embeddings[utt_id] for utt_id in ids], axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def score_mean_trials(
    trial_jsonl: str | Path,
    original_embeddings: Mapping[str, np.ndarray],
    anonymized_embeddings: Mapping[str, np.ndarray] | None,
    condition: str,
    n: int,
    output_csv: str | Path,
) -> dict[str, object]:
    if condition not in {"O-O", "O-A", "A-A"}:
        raise ValueError(f"未知 condition: {condition}")
    if condition != "O-O" and anonymized_embeddings is None:
        raise ValueError(f"{condition} 需要 anonymized embeddings")
    enroll_source = original_embeddings if condition != "A-A" else anonymized_embeddings
    target_source = original_embeddings if condition == "O-O" else anonymized_embeddings
    assert enroll_source is not None and target_source is not None

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels: list[int] = []
    scores: list[float] = []
    with Path(trial_jsonl).open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=["trial_id", "label", "score"])
        writer.writeheader()
        for line in source:
            trial = json.loads(line)
            enroll = _aggregate(trial["enroll_utt_ids"][:n], enroll_source)
            target = _aggregate(trial["target_utt_ids"][:n], target_source)
            score = float(np.dot(enroll, target))
            label = int(trial["label"])
            labels.append(label)
            scores.append(score)
            writer.writerow({"trial_id": trial["trial_id"], "label": label, "score": score})
    eer, threshold = compute_eer(np.asarray(labels), np.asarray(scores))
    result = {
        "condition": condition,
        "aggregation": "mean",
        "n": n,
        "trials": len(labels),
        "eer": eer,
        "eer_percent": eer * 100.0,
        "threshold": threshold,
        "score_file": str(output_path.resolve()),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def score_session_trials(
    trial_jsonl: str | Path,
    original_embeddings: Mapping[str, np.ndarray],
    anonymized_embeddings: Mapping[str, np.ndarray] | None,
    condition: str,
    n: int,
    output_csv: str | Path,
) -> dict[str, object]:
    """Score the SAAR main protocol.

    Enrollment is fixed for the entire N sweep.  Only the target-side prefix is
    changed, which isolates the effect of accumulating anonymous observations.
    """

    if condition not in {"O-O", "O-A", "A-A"}:
        raise ValueError(f"unknown condition: {condition}")
    if condition != "O-O" and anonymized_embeddings is None:
        raise ValueError(f"{condition} requires anonymized embeddings")
    enroll_source = original_embeddings if condition != "A-A" else anonymized_embeddings
    target_source = original_embeddings if condition == "O-O" else anonymized_embeddings
    assert enroll_source is not None and target_source is not None

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels: list[int] = []
    scores: list[float] = []
    seed: int | None = None
    enrollment_sizes: set[int] = set()
    with Path(trial_jsonl).open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "trial_id", "seed", "label", "n", "enrollment_n", "score",
                "enroll_session_id", "target_session_id",
            ],
        )
        writer.writeheader()
        for line_number, line in enumerate(source, start=1):
            trial = json.loads(line)
            target_ids = list(map(str, trial["target_utt_ids"]))
            if len(target_ids) < n:
                raise ValueError(f"trial line {line_number} has fewer than N={n} target utterances")
            trial_seed = int(trial["seed"])
            if seed is None:
                seed = trial_seed
            elif seed != trial_seed:
                raise ValueError("a score file must contain exactly one sampling seed")
            enroll_ids = list(map(str, trial["enroll_utt_ids"]))
            enrollment_sizes.add(len(enroll_ids))
            enroll = _aggregate(enroll_ids, enroll_source)
            target = _aggregate(target_ids[:n], target_source)
            score = float(np.dot(enroll, target))
            label = int(trial["label"])
            labels.append(label)
            scores.append(score)
            writer.writerow({
                "trial_id": trial["trial_id"],
                "seed": trial_seed,
                "label": label,
                "n": n,
                "enrollment_n": len(enroll_ids),
                "score": score,
                "enroll_session_id": trial["enroll_session_id"],
                "target_session_id": trial["target_session_id"],
            })
    if seed is None:
        raise ValueError("trial file is empty")
    if len(enrollment_sizes) != 1:
        raise ValueError("fixed enrollment size is inconsistent across trials")
    eer, threshold = compute_eer(np.asarray(labels), np.asarray(scores))
    result = {
        "protocol": "fixed-original-enrollment/increasing-anonymized-target",
        "condition": condition,
        "aggregation": "mean",
        "seed": seed,
        "n": n,
        "enrollment_n": next(iter(enrollment_sizes)),
        "enrollment_fixed_across_n": True,
        "trials": len(labels),
        "eer": eer,
        "eer_percent": eer * 100.0,
        "threshold": threshold,
        "score_file": str(output_path.resolve()),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def compute_pcs(
    trial_jsonl_paths: Sequence[str | Path],
    anonymized_embeddings: Mapping[str, np.ndarray],
    output_csv: str | Path,
) -> dict[str, object]:
    """Compute pseudo consistency as mean pairwise cosine within each session."""

    by_session: dict[str, set[str]] = {}
    for path in trial_jsonl_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                trial = json.loads(line)
                session_id = str(trial["target_session_id"])
                by_session.setdefault(session_id, set()).update(
                    map(str, trial["target_utt_ids"])
                )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values: list[float] = []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "utterances", "pcs"])
        writer.writeheader()
        for session_id, utterance_ids in sorted(by_session.items()):
            ordered_ids = sorted(utterance_ids)
            if len(ordered_ids) < 2:
                continue
            missing = [utt_id for utt_id in ordered_ids if utt_id not in anonymized_embeddings]
            if missing:
                raise KeyError(
                    f"session {session_id} is missing {len(missing)} embeddings, e.g. {missing[0]}"
                )
            matrix = np.stack([anonymized_embeddings[utt_id] for utt_id in ordered_ids])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1.0e-12)
            vector_sum = matrix.sum(axis=0)
            pair_sum = (float(np.dot(vector_sum, vector_sum)) - len(matrix)) / 2.0
            pair_count = len(matrix) * (len(matrix) - 1) / 2.0
            pcs = pair_sum / pair_count
            values.append(pcs)
            writer.writerow({
                "session_id": session_id,
                "utterances": len(matrix),
                "pcs": pcs,
            })
    if not values:
        raise ValueError("PCS requires at least one session with two embeddings")
    result = {
        "definition": "mean_session_mean_pairwise_cosine",
        "sessions": len(values),
        "pcs": float(np.mean(values)),
        "pcs_std_across_sessions": float(np.std(values)),
        "per_session_file": str(output_path.resolve()),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def summarize_privacy_curve(
    metric_json_paths: Sequence[str | Path],
    output_csv: str | Path,
    *,
    system: str,
    attacker: str,
    checkpoint: str,
    git_commit: str,
) -> dict[str, object]:
    records = []
    for path in metric_json_paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        record["metrics_file"] = str(Path(path).resolve())
        records.append(record)
    by_seed: dict[int, dict[int, dict[str, object]]] = {}
    for record in records:
        seed = int(record["seed"])
        n = int(record["n"])
        if n in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate metric for seed={seed}, N={n}")
        by_seed[seed][n] = record

    expected_n = sorted({int(record["n"]) for record in records})
    if 1 not in expected_n:
        raise ValueError("privacy curve requires N=1")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system", "attacker", "seed", "N", "eer", "eer_percent", "eer1",
        "delta_eer", "relative_degradation", "slope_beta", "abs_slope_beta",
        "checkpoint", "git_commit", "metrics_file", "score_file",
    ]
    rows: list[dict[str, object]] = []
    slopes: list[float] = []
    for seed, curve in sorted(by_seed.items()):
        if sorted(curve) != expected_n:
            raise ValueError(f"seed {seed} does not have the complete N sweep {expected_n}")
        eer1 = float(curve[1]["eer"])
        eers = np.asarray([float(curve[n]["eer"]) for n in expected_n])
        slope = float(np.polyfit(np.log2(np.asarray(expected_n, dtype=np.float64)), eers, 1)[0])
        slopes.append(slope)
        for n in expected_n:
            record = curve[n]
            eer = float(record["eer"])
            degradation = eer1 - eer
            rows.append({
                "system": system,
                "attacker": attacker,
                "seed": seed,
                "N": n,
                "eer": eer,
                "eer_percent": eer * 100.0,
                "eer1": eer1,
                "delta_eer": degradation,
                "relative_degradation": (
                    math.nan if eer1 == 0.0 else degradation / eer1 * 100.0
                ),
                "slope_beta": slope,
                "abs_slope_beta": abs(slope),
                "checkpoint": checkpoint,
                "git_commit": git_commit,
                "metrics_file": str(record.get("metrics_file", "")),
                "score_file": str(record["score_file"]),
            })
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for n in expected_n:
        values = np.asarray([float(by_seed[seed][n]["eer"]) for seed in sorted(by_seed)])
        aggregate.append({
            "n": n,
            "eer_mean": float(values.mean()),
            "eer_std": float(values.std()),
        })
    result = {
        "system": system,
        "attacker": attacker,
        "seeds": sorted(by_seed),
        "n_values": expected_n,
        "per_n": aggregate,
        "slope_beta_mean": float(np.mean(slopes)),
        "abs_slope_beta_mean": float(np.mean(np.abs(slopes))),
        "privacy_summary_csv": str(output_path.resolve()),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
