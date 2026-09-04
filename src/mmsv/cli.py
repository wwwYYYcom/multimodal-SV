from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .aggregation import QueryAttention, mean_pool
from .anonymization import anonymize_plan, build_anonymization_plan
from .config import load_config, require_path
from .data.fisher import build_manifest
from .data.librispeech import build_librispeech_pool
from .data.session_trials import build_session_trial_sets
from .data.splits import split_speakers
from .data.trials import build_trials, validate_trials
from .metrics import (
    compute_pcs,
    load_embeddings,
    score_mean_trials,
    score_session_trials,
    summarize_privacy_curve,
)
from .models import ECAPABackend
from .plotting import plot_privacy_curves
from .train import extract_embeddings, train_audio


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _prepare_fisher(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = config["data"]
    _print(build_manifest(
        require_path(data["fisher_audio_root"], "fisher_audio_root"),
        require_path(data["fisher_transcript_root"], "fisher_transcript_root"),
        require_path(data["fisher_calldata"], "fisher_calldata"),
        args.output,
        float(data.get("min_segment_seconds", 1.0)),
        args.max_calls,
    ))


def _split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    values = config["split"]
    _print(split_speakers(
        args.manifest,
        args.output,
        int(values["train_speakers"]),
        int(values["validation_speakers"]),
        int(values["evaluation_speakers"]),
        int(config["seed"]),
        bool(values["require_exact_counts"]),
        int(values.get("evaluation_min_calls", 2)),
        int(values.get("validation_min_calls", 2)),
        int(values.get("max_n", 15)),
    ))


def _trials(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    values = config["trials"]
    prefix = "validation" if args.split == "validation" else "evaluation"
    result = build_trials(
        args.manifest,
        args.splits,
        args.output,
        args.split,
        values["n_values"],
        int(values[f"{prefix}_target_per_speaker"]),
        int(values[f"{prefix}_nontarget_per_speaker"]),
        int(config["seed"]),
    )
    result["validation"] = validate_trials(args.output, values["n_values"], args.manifest)
    _print(result)


def _libri(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    roots = [require_path(value, "librispeech_root") for value in config["data"]["librispeech_roots"] if value]
    _print(build_librispeech_pool(roots, args.output, args.min_duration))


def _score(args: argparse.Namespace) -> None:
    original = load_embeddings(args.original_embeddings)
    anonymized = load_embeddings(args.anonymized_embeddings) if args.anonymized_embeddings else None
    _print(score_mean_trials(args.trials, original, anonymized, args.condition, args.n, args.output))


def _score_session(args: argparse.Namespace) -> None:
    original = load_embeddings(args.original_embeddings)
    anonymized = load_embeddings(args.anonymized_embeddings) if args.anonymized_embeddings else None
    _print(score_session_trials(
        args.trials, original, anonymized, args.condition, args.n, args.output
    ))


def _pcs(args: argparse.Namespace) -> None:
    _print(compute_pcs(
        args.trials,
        load_embeddings(args.anonymized_embeddings),
        args.output,
    ))


def _summarize_privacy(args: argparse.Namespace) -> None:
    _print(summarize_privacy_curve(
        args.metrics,
        args.output,
        system=args.system,
        attacker=args.attacker,
        checkpoint=args.checkpoint,
        git_commit=args.git_commit,
    ))


def _plot_privacy(args: argparse.Namespace) -> None:
    _print(plot_privacy_curves(args.input, args.output))


def _model_smoke(_: argparse.Namespace) -> None:
    torch.manual_seed(7)
    backend = ECAPABackend(input_dim=16, channels=32, mfa_channels=64, embedding_dim=24)
    features = torch.randn(2, 40, 16)
    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[1, 31:] = False
    embeddings = backend(features, mask)
    utterances = torch.randn(2, 5, 24)
    query = QueryAttention(dim=24, heads=4, temperature=0.3)(utterances)
    mean = mean_pool(utterances)
    _print({
        "ecapa_shape": list(embeddings.shape),
        "query_shape": list(query.shape),
        "mean_shape": list(mean.shape),
        "normalized": bool(torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5)),
    })


def _plan_anonymization(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _print(build_anonymization_plan(
        args.manifest,
        args.reference_pool,
        args.output,
        args.audio_output_root,
        int(config["seed"]),
        args.trials,
        args.limit,
        args.splits,
        args.split_name,
        args.one_per_call_side,
        args.reference_mapping,
        args.mapping_output,
        args.trial_role,
    ))


def _prepare_saar_baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    evaluation = config["eval"]
    pseudo = config["pseudo"]
    output_root = Path(args.output_root)
    manifest_root = output_root / "manifests"
    allowed_utterance_ids: set[str] | None = None
    if args.allowed_embeddings:
        archive = np.load(args.allowed_embeddings, allow_pickle=False)
        allowed_utterance_ids = set(archive["utt_ids"].astype(str).tolist())
    trial_audit = build_session_trial_sets(
        args.manifest,
        args.splits,
        manifest_root,
        args.split,
        evaluation["n_list"],
        int(evaluation["enrollment_n"]),
        evaluation["seeds"],
        allowed_utterance_ids,
    )
    plan_audit = build_anonymization_plan(
        args.manifest,
        args.reference_pool,
        manifest_root / "session_baseline_anonymization_plan.csv",
        args.audio_output_root,
        int(config["experiment"]["seed"]),
        manifest_root / "all_seeds.jsonl",
        reference_mapping=str(pseudo["mode"]).replace("_fixed", ""),
        mapping_output_json=manifest_root / "pseudo_mapping.json",
        trial_role="target",
    )
    _print({
        "phase": "SAAR Phase 1/2 preparation",
        "trial_audit": trial_audit,
        "anonymization_plan_audit": plan_audit,
    })


def _anonymize(args: argparse.Namespace) -> None:
    _print(anonymize_plan(
        args.plan,
        args.output_manifest,
        args.streamvoice_root,
        args.streamvoice_config,
        args.checkpoint,
        args.sample_rate,
        args.delay,
        args.alpha,
        args.sph2pipe,
        args.limit,
        args.start_index,
        args.compile_ar,
        args.compile_encoder,
        args.compile_decoder,
        args.fp16,
        args.max_source_chunk_seconds,
    ))


def _train(args: argparse.Namespace) -> None:
    _print(train_audio(
        load_config(args.config), args.manifest, args.splits, args.output_dir,
        args.sph2pipe, args.resume, args.init_from, args.max_steps
    ))


def _extract(args: argparse.Namespace) -> None:
    _print(extract_embeddings(
        args.checkpoint,
        args.manifest,
        args.output,
        args.sample_rate,
        args.sph2pipe,
        args.limit,
        args.trials,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmsv", description="多语句多模态 ASV 论文复现")
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare-fisher", help="构建 Fisher utterance manifest")
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--max-calls", type=int)
    command.set_defaults(func=_prepare_fisher)

    command = subparsers.add_parser("split-speakers", help="生成 speaker-disjoint split")
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_split)

    command = subparsers.add_parser("build-trials", help="生成固定 nested multi-utterance trials")
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--splits", required=True)
    command.add_argument("--split", choices=["validation", "evaluation"], required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_trials)

    command = subparsers.add_parser("build-libri-pool", help="筛选 >4 秒的匿名化 reference pool")
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--min-duration", type=float, default=4.0)
    command.set_defaults(func=_libri)

    command = subparsers.add_parser("model-smoke", help="不下载权重的模型 shape/归一化冒烟测试")
    command.set_defaults(func=_model_smoke)

    command = subparsers.add_parser("plan-anonymization", help="生成 Fisher -> clean-360 确定性匿名化计划")
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--reference-pool", required=True)
    command.add_argument("--audio-output-root", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--trials", help="只规划该 trial JSONL 引用的 utterance")
    command.add_argument("--splits", help="按 speaker split 过滤；不能与 --trials 同时使用")
    command.add_argument("--split-name", choices=["train", "validation", "evaluation"])
    command.add_argument(
        "--one-per-call-side",
        action="store_true",
        help="每个 call-side 确定性选择一条 source；用于磁盘受限的 semi-informed 训练",
    )
    command.add_argument(
        "--reference-mapping",
        choices=["utterance", "session"],
        default="utterance",
        help="reference selection unit; session uses Fisher call_id+channel",
    )
    command.add_argument("--mapping-output", help="persist session_id -> pseudo mapping JSON")
    command.add_argument(
        "--trial-role",
        choices=["all", "enroll", "target"],
        default="all",
        help="when --trials is used, select which side must be anonymized",
    )
    command.add_argument("--limit", type=int)
    command.set_defaults(func=_plan_anonymization)

    command = subparsers.add_parser("anonymize-streamvoice", help="按计划运行可断点续跑的 StreamVoiceAnon")
    command.add_argument("--plan", required=True)
    command.add_argument("--output-manifest", required=True)
    command.add_argument("--streamvoice-root", required=True)
    command.add_argument("--streamvoice-config", default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    command.add_argument("--checkpoint", default="pretrained_checkpoints/dual_ar_delay_0_8.pth")
    command.add_argument("--sample-rate", type=int, default=16000)
    command.add_argument("--delay", type=int, default=2)
    command.add_argument("--alpha", type=float, default=1.0)
    command.add_argument("--sph2pipe")
    command.add_argument("--limit", type=int)
    command.add_argument("--start-index", type=int, default=0)
    command.add_argument("--compile-ar", action="store_true")
    command.add_argument("--compile-encoder", action="store_true")
    command.add_argument("--compile-decoder", action="store_true")
    command.add_argument("--fp16", action="store_true")
    command.add_argument(
        "--max-source-chunk-seconds",
        type=float,
        default=30.0,
        help="对更长 source 做等长分块后分别匿名化并拼接；避免上游 KV cache 越界",
    )
    command.set_defaults(func=_anonymize)

    command = subparsers.add_parser(
        "prepare-saar-baseline",
        help="build session-fixed pseudo mapping and fixed-enrollment session trials",
    )
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--splits", required=True)
    command.add_argument("--reference-pool", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--audio-output-root", required=True)
    command.add_argument(
        "--allowed-embeddings",
        help="restrict protocol to utterances already present in this NPZ",
    )
    command.add_argument("--split", choices=["validation", "evaluation"], default="evaluation")
    command.set_defaults(func=_prepare_saar_baseline)

    command = subparsers.add_parser("train-audio", help="训练 WavLM-Large + ECAPA-TDNN")
    command.add_argument("--config", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--splits", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--sph2pipe")
    command.add_argument("--resume")
    command.add_argument(
        "--init-from",
        help="仅加载模型/分类头并重置 optimizer；用于 lazy -> semi-informed",
    )
    command.add_argument("--max-steps", type=int)
    command.set_defaults(func=_train)

    command = subparsers.add_parser("extract-embeddings", help="从 checkpoint 提取 utterance embeddings")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--manifest", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--sample-rate", type=int, default=16000)
    command.add_argument("--sph2pipe")
    command.add_argument("--limit", type=int)
    command.add_argument(
        "--trials",
        help="只提取该 JSONL trial 文件引用的 utterance；正式评测推荐使用",
    )
    command.set_defaults(func=_extract)

    command = subparsers.add_parser("score-mean", help="按 O-O/O-A/A-A 条件计算 mean pooling EER")
    command.add_argument("--trials", required=True)
    command.add_argument("--original-embeddings", required=True)
    command.add_argument("--anonymized-embeddings")
    command.add_argument("--condition", choices=["O-O", "O-A", "A-A"], required=True)
    command.add_argument("--n", type=int, choices=[1, 2, 5, 10, 15], required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_score)

    command = subparsers.add_parser(
        "score-session",
        help="score fixed-enrollment/increasing-target session trials",
    )
    command.add_argument("--trials", required=True)
    command.add_argument("--original-embeddings", required=True)
    command.add_argument("--anonymized-embeddings")
    command.add_argument("--condition", choices=["O-O", "O-A", "A-A"], required=True)
    command.add_argument("--n", type=int, choices=[1, 2, 5, 10, 15], required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_score_session)

    command = subparsers.add_parser(
        "compute-pcs",
        help="compute session pseudo consistency from anonymous embeddings",
    )
    command.add_argument("--trials", nargs="+", required=True)
    command.add_argument("--anonymized-embeddings", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_pcs)

    command = subparsers.add_parser(
        "summarize-privacy",
        help="summarize EER, delta EER, relative degradation and slope beta",
    )
    command.add_argument("--metrics", nargs="+", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--system", required=True)
    command.add_argument("--attacker", default="lazy-informed WavLM-ECAPA mean")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--git-commit", required=True)
    command.set_defaults(func=_summarize_privacy)

    command = subparsers.add_parser(
        "plot-privacy",
        help="plot mean and seed variation for one or more privacy summaries",
    )
    command.add_argument("--input", nargs="+", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_plot_privacy)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
