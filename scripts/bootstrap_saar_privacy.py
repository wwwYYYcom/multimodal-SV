from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mmsv.metrics import compute_eer


def _load_scores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    trial_ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trial_ids.append(row["trial_id"])
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return trial_ids, np.asarray(labels), np.asarray(scores)


def bootstrap_privacy(
    score_root: Path,
    output_csv: Path,
    *,
    seeds: list[int],
    n_values: list[int],
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    score_data: dict[int, dict[int, np.ndarray]] = {}
    labels_for_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        reference_ids: list[str] | None = None
        reference_labels: np.ndarray | None = None
        score_data[seed] = {}
        for n in n_values:
            path = score_root / f"oa_mean_N{n}_seed{seed}.csv"
            trial_ids, labels, scores = _load_scores(path)
            if reference_ids is None:
                reference_ids, reference_labels = trial_ids, labels
            elif trial_ids != reference_ids or not np.array_equal(labels, reference_labels):
                raise ValueError(f"N sweep is not paired for seed={seed}, N={n}")
            score_data[seed][n] = scores
        assert reference_labels is not None
        labels_for_seed[seed] = reference_labels

    rng = np.random.default_rng(bootstrap_seed)
    samples = {n: np.empty(replicates, dtype=np.float64) for n in n_values}
    delta15 = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        replicate_eers = {n: [] for n in n_values}
        for seed in seeds:
            labels = labels_for_seed[seed]
            positive = np.flatnonzero(labels == 1)
            negative = np.flatnonzero(labels == 0)
            sampled = np.concatenate([
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ])
            sampled_labels = labels[sampled]
            for n in n_values:
                eer, _ = compute_eer(sampled_labels, score_data[seed][n][sampled])
                replicate_eers[n].append(eer)
        for n in n_values:
            samples[n][replicate] = float(np.mean(replicate_eers[n]))
        delta15[replicate] = samples[1][replicate] - samples[15][replicate]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in n_values:
        values = samples[n] * 100.0
        rows.append({
            "N": n,
            "eer_bootstrap_mean_percent": float(values.mean()),
            "eer_ci95_low_percent": float(np.quantile(values, 0.025)),
            "eer_ci95_high_percent": float(np.quantile(values, 0.975)),
        })
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    delta_values = delta15 * 100.0
    result = {
        "method": "paired_stratified_trial_bootstrap_then_mean_across_seeds",
        "seeds": seeds,
        "n_values": n_values,
        "replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "per_n": rows,
        "delta_eer15_bootstrap_mean_percentage_points": float(delta_values.mean()),
        "delta_eer15_ci95_low_percentage_points": float(np.quantile(delta_values, 0.025)),
        "delta_eer15_ci95_high_percentage_points": float(np.quantile(delta_values, 0.975)),
        "output_csv": str(output_csv.resolve()),
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap SAAR EER curves and Delta-EER15")
    parser.add_argument("--score-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 2, 5, 10, 15])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    result = bootstrap_privacy(
        args.score_root,
        args.output,
        seeds=args.seeds,
        n_values=args.n_values,
        replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
