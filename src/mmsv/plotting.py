from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def plot_privacy_curves(
    summary_csv_paths: Sequence[str | Path],
    output_png: str | Path,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in summary_csv_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                curves[row["system"]][int(row["N"])].append(float(row["eer_percent"]))
    if not curves:
        raise ValueError("no privacy summary rows were found")

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for system, by_n in sorted(curves.items()):
        n_values = sorted(by_n)
        means = np.asarray([np.mean(by_n[n]) for n in n_values])
        stds = np.asarray([np.std(by_n[n]) for n in n_values])
        axis.errorbar(
            n_values,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=3,
            label=system,
        )
    axis.set_xlabel("Anonymous target utterances (N)")
    axis.set_ylabel("O-A EER (%)")
    axis.set_xticks(sorted({n for by_n in curves.values() for n in by_n}))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "output": str(output_path.resolve()),
        "systems": sorted(curves),
        "inputs": [str(Path(path).resolve()) for path in summary_csv_paths],
    }
