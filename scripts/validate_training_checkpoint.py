from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="验证正式训练 checkpoint 是否完整")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-last-epoch", required=True, type=int)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "epoch": int(state["epoch"]),
        "epoch_complete": bool(state.get("epoch_complete", True)),
        "batch_in_epoch": int(state.get("batch_in_epoch", 0)),
        "global_step": int(state["global_step"]),
    }
    print(json.dumps(metadata, ensure_ascii=False))
    if metadata["epoch"] != args.expected_last_epoch or not metadata["epoch_complete"]:
        raise SystemExit(f"拒绝评测：训练 checkpoint 不完整（{metadata}）")


if __name__ == "__main__":
    main()
