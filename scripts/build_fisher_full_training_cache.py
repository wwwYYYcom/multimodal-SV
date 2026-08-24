from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from mmsv.audio import read_segment
from mmsv.config import load_config
from mmsv.train import FisherTrainingDataset, _epoch_indices


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="缓存 Fisher Part 1 train split 的全部 utterance（16 kHz PCM16 FLAC）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-cache-dir")
    parser.add_argument("--sph2pipe")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    reuse_dir = (
        Path(args.reuse_cache_dir).expanduser().resolve() if args.reuse_cache_dir else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = FisherTrainingDataset(
        args.manifest,
        args.splits,
        "train",
        int(config["sample_rate"]),
        float(config["crop_seconds"]),
        int(config["seed"]),
        args.sph2pipe,
        sampling_mode="all_utterances",
        short_utterance_mode="repeat",
    )

    indices: list[int] | range = range(len(dataset))
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError("--limit 必须位于 [1, train_utterances]")
        indices = _epoch_indices(len(dataset), int(config["seed"]), args.epoch, True)[:args.limit]

    generated = 0
    linked = 0
    skipped = 0
    for index in tqdm(indices, desc="Fisher full cache", unit="utt"):
        row = dataset.selected_row(index)
        utt_id = row["utt_id"]
        destination = dataset.cache_path(output_dir, utt_id)
        if destination.is_file() and destination.stat().st_size > 0:
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if reuse_dir is not None:
            reusable = dataset.cache_path(reuse_dir, utt_id)
            if reusable.is_file() and reusable.stat().st_size > 0:
                try:
                    os.link(reusable, destination)
                except OSError:
                    pass
                else:
                    linked += 1
                    continue
        waveform = read_segment(row, dataset.sample_rate, args.sph2pipe)
        temporary = destination.with_suffix(".tmp.flac")
        sf.write(temporary, waveform, dataset.sample_rate, format="FLAC", subtype="PCM_16")
        temporary.replace(destination)
        generated += 1

    if generated + linked + skipped != len(indices):
        raise RuntimeError("缓存计数与本次目标 utterance 数不一致")
    audit = {
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "seconds": time.time() - started,
        "train_utterances": len(dataset),
        "target_utterances": len(indices),
        "epoch_order": args.epoch if args.limit is not None else None,
        "generated": generated,
        "hardlinked": linked,
        "skipped_existing": skipped,
        "sample_rate": dataset.sample_rate,
        "format": "FLAC/PCM_16",
        "output_dir": str(output_dir),
        "reuse_cache_dir": None if reuse_dir is None else str(reuse_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(Path(args.manifest)),
        "splits": str(Path(args.splits).resolve()),
        "splits_sha256": sha256(Path(args.splits)),
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
