from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bootstrap_saar_privacy import bootstrap_privacy
from mmsv.metrics import (
    compute_pcs,
    load_embeddings,
    score_session_trials,
    summarize_privacy_curve,
)
from mmsv.plotting import plot_privacy_curves


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SAAR session-fixed baseline and make the Gate 1 decision"
    )
    parser.add_argument("--trials-dir", required=True, type=Path)
    parser.add_argument("--original-embeddings", required=True, type=Path)
    parser.add_argument("--anonymized-embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--control-summary", type=Path)
    parser.add_argument("--gate-min-delta-eer15-pp", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    git_commit = _git_commit(project_root)
    original = load_embeddings(args.original_embeddings)
    anonymized = load_embeddings(args.anonymized_embeddings)
    score_root = args.output_root / "scores"
    metric_root = args.output_root / "metrics"
    figure_root = args.output_root / "figures"
    for path in (score_root, metric_root, figure_root):
        path.mkdir(parents=True, exist_ok=True)

    metric_paths: list[Path] = []
    n_values = [1, 2, 5, 10, 15]
    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        trial_path = args.trials_dir / f"sampling_seed_{seed}.jsonl"
        for n in n_values:
            score_path = score_root / f"oa_mean_N{n}_seed{seed}.csv"
            score_session_trials(
                trial_path,
                original,
                anonymized,
                "O-A",
                n,
                score_path,
            )
            metric_paths.append(score_path.with_suffix(".metrics.json"))

    summary_path = metric_root / "privacy_summary.csv"
    summary = summarize_privacy_curve(
        metric_paths,
        summary_path,
        system="Session-fixed baseline",
        attacker="lazy-informed WavLM-ECAPA mean",
        checkpoint=str(args.checkpoint.resolve()),
        git_commit=git_commit,
    )
    pcs = compute_pcs(
        [args.trials_dir / "all_seeds.jsonl"],
        anonymized,
        metric_root / "pcs_summary.csv",
    )
    bootstrap = bootstrap_privacy(
        score_root,
        metric_root / "bootstrap_ci.csv",
        seeds=seeds,
        n_values=n_values,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    per_n = {int(row["n"]): row for row in summary["per_n"]}
    delta_eer15_pp = (
        float(per_n[1]["eer_mean"]) - float(per_n[15]["eer_mean"])
    ) * 100.0
    ci_excludes_zero = (
        float(bootstrap["delta_eer15_ci95_low_percentage_points"]) > 0.0
    )
    gate_passed = (
        delta_eer15_pp >= args.gate_min_delta_eer15_pp and ci_excludes_zero
    )
    gate = {
        "gate": "Gate 1: aggregation vulnerability exists before SAAR training",
        "criterion": (
            "mean O-A EER(N=1) - mean O-A EER(N=15) >= "
            f"{args.gate_min_delta_eer15_pp:.6f} percentage points and paired "
            "stratified-bootstrap 95% CI excludes zero"
        ),
        "delta_eer15_percentage_points": delta_eer15_pp,
        "delta_eer15_ci95_low_percentage_points": bootstrap[
            "delta_eer15_ci95_low_percentage_points"
        ],
        "delta_eer15_ci95_high_percentage_points": bootstrap[
            "delta_eer15_ci95_high_percentage_points"
        ],
        "bootstrap_ci_excludes_zero": ci_excludes_zero,
        "passed": gate_passed,
        "next_action": (
            "proceed_to_saar_mvp_training"
            if gate_passed
            else "stop_and_recheck_protocol_or_revise_hypothesis"
        ),
    }
    gate_path = metric_root / "gate_1.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_inputs = [summary_path]
    if args.control_summary is not None and args.control_summary.is_file():
        plot_inputs.insert(0, args.control_summary)
    try:
        plot = plot_privacy_curves(plot_inputs, figure_root / "eer_vs_n.png")
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        plot = {
            "status": "deferred_to_local",
            "reason": "matplotlib is not installed on this host",
            "command": (
                "python -m mmsv.cli plot-privacy --input "
                + " ".join(map(str, plot_inputs))
                + f" --output {figure_root / 'eer_vs_n.png'}"
            ),
        }

    result = {
        "git_commit": git_commit,
        "privacy": summary,
        "pcs": pcs,
        "bootstrap": bootstrap,
        "gate_1": gate,
        "plot": plot,
    }
    output = args.output_root / "evaluation_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
