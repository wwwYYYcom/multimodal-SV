from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

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
