from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .aggregation import QueryAttention, mean_pool
from .config import load_config, require_path
from .data.fisher import build_manifest
from .data.librispeech import build_librispeech_pool
from .data.splits import split_speakers
from .data.trials import build_trials, validate_trials
from .metrics import load_embeddings, score_mean_trials
from .models import ECAPABackend
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
    command.add_argument("--n", type=int, choices=[5, 10, 15], required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
