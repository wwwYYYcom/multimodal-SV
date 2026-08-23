from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from mmsv.audio import read_segment
from mmsv.config import load_config
from mmsv.train import FisherTrainingDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="缓存正式训练各 epoch 会抽到的 Fisher utterance（16 kHz PCM16 FLAC）"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sph2pipe")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    epochs = int(config["train"]["epochs"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = FisherTrainingDataset(
        args.manifest,
        args.splits,
        "train",
        int(config["sample_rate"]),
        float(config["crop_seconds"]),
        int(config["seed"]),
        args.sph2pipe,
    )

    planned: dict[str, dict[str, str]] = {}
    for index in range(len(dataset)):
        for epoch in range(epochs):
            row = dataset.selected_row(index, epoch)
            planned.setdefault(row["utt_id"], row)

    generated = 0
    skipped = 0
    seen: set[str] = set()
    for index in tqdm(range(len(dataset)), desc="Fisher cache", unit="call-side"):
        for epoch in range(epochs):
            row = dataset.selected_row(index, epoch)
            utt_id = row["utt_id"]
            if utt_id in seen:
                continue
            seen.add(utt_id)
            destination = dataset.cache_path(output_dir, utt_id)
            if destination.is_file() and destination.stat().st_size > 0:
                skipped += 1
                continue
            waveform = read_segment(row, dataset.sample_rate, args.sph2pipe)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.flac")
            sf.write(temporary, waveform, dataset.sample_rate, format="FLAC", subtype="PCM_16")
            temporary.replace(destination)
            generated += 1

    if seen != set(planned):
        raise RuntimeError("缓存遍历结果与预计算计划不一致")
    audit = {
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "seconds": time.time() - started,
        "epochs": epochs,
        "call_sides": len(dataset),
        "unique_utterances": len(planned),
        "generated": generated,
        "skipped_existing": skipped,
        "sample_rate": dataset.sample_rate,
        "format": "FLAC/PCM_16",
        "output_dir": str(output_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(Path(args.manifest)),
        "splits": str(Path(args.splits).resolve()),
        "splits_sha256": sha256(Path(args.splits)),
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
