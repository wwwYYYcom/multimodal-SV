from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .audio import crop_or_pad, crop_or_repeat, read_audio, read_segment
from .data.fisher import read_manifest
from .models import AAMSoftmaxLoss, WavLMECAPA


def _read_split_map(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["speaker_id"]: row["split"] for row in csv.DictReader(handle)}


class FisherTrainingDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path,
        split_name: str,
        sample_rate: int,
        crop_seconds: float,
        seed: int,
        sph2pipe: str | None,
        segment_cache_dir: str | Path | None = None,
        sampling_mode: str = "one_per_call_side",
        short_utterance_mode: str = "zero_pad",
    ):
        split_for = _read_split_map(split_path)
        rows = [
            row for row in read_manifest(manifest_path)
            if split_for.get(row["speaker_id"]) == split_name
        ]
        if not rows:
            raise ValueError(f"{split_name} split 没有 utterance")
        speakers = sorted({row["speaker_id"] for row in rows})
        self.speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}
        if sampling_mode not in {"one_per_call_side", "all_utterances"}:
            raise ValueError(f"未知 sampling_mode: {sampling_mode}")
        if short_utterance_mode not in {"zero_pad", "repeat"}:
            raise ValueError(f"未知 short_utterance_mode: {short_utterance_mode}")
        self.sampling_mode = sampling_mode
        self.short_utterance_mode = short_utterance_mode
        self.rows: list[dict[str, str]] | None = None
        self.groups: list[list[dict[str, str]]] | None = None
        if sampling_mode == "all_utterances":
            self.rows = rows
        else:
            by_call_side: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_call_side[(row["audio_path"], row["channel"])].append(row)
            # 每 epoch 每个 call side 采一条 turn；A/B 相邻可复用 Shorten 解码缓存。
            by_audio: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
            for (audio_path, channel), group in by_call_side.items():
                by_audio[audio_path].append((channel, group))
            audio_paths = sorted(by_audio)
            random.Random(seed).shuffle(audio_paths)
            self.groups = []
            for audio_path in audio_paths:
                self.groups.extend(
                    group for _, group in sorted(by_audio[audio_path], key=lambda item: item[0])
                )
        self.sample_rate = sample_rate
        self.crop_samples = round(sample_rate * crop_seconds)
        self.seed = seed
        self.epoch = 0
        self.sph2pipe = sph2pipe
        self.segment_cache_dir = (
            Path(segment_cache_dir).expanduser().resolve() if segment_cache_dir else None
        )

    def __len__(self) -> int:
        return len(self.rows) if self.rows is not None else len(self.groups or [])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def selected_row(self, index: int, epoch: int | None = None) -> dict[str, str]:
        if self.rows is not None:
            return self.rows[index]
        assert self.groups is not None
        selected_epoch = self.epoch if epoch is None else int(epoch)
        group = self.groups[index]
        rng = random.Random(self.seed + selected_epoch * 1_000_003 + index)
        return group[rng.randrange(len(group))]

    @staticmethod
    def cache_path(root: str | Path, utt_id: str) -> Path:
        import hashlib

        shard = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()[:2]
        return Path(root) / shard / f"{utt_id}.flac"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        if self.rows is None:
            assert self.groups is not None
            group = self.groups[index]
            row = group[rng.randrange(len(group))]
        else:
            row = self.rows[index]
        if self.segment_cache_dir is None:
            waveform = read_segment(row, self.sample_rate, self.sph2pipe)
        else:
            cache_path = self.cache_path(self.segment_cache_dir, row["utt_id"])
            if not cache_path.is_file():
                raise FileNotFoundError(f"训练片段缓存缺失: {cache_path}")
            waveform, sample_rate = read_audio(cache_path)
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"训练片段缓存采样率错误: {cache_path} ({sample_rate} != {self.sample_rate})"
                )
        start = rng.randrange(max(1, waveform.size - self.crop_samples + 1))
        if self.short_utterance_mode == "repeat":
            waveform = crop_or_repeat(waveform, self.crop_samples, start)
        else:
            waveform = crop_or_pad(waveform, self.crop_samples, start)
        return torch.from_numpy(waveform), self.speaker_to_index[row["speaker_id"]]


def _save_checkpoint(
    path: Path,
    model: WavLMECAPA,
    classifier: AAMSoftmaxLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    speaker_to_index: dict[str, int],
    settings: dict[str, Any],
    *,
    epoch_complete: bool = True,
    batch_in_epoch: int = 0,
    running_loss: float = 0.0,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    model_state = model.state_dict()
    if model.frozen:
        model_state = {
            name: value for name, value in model_state.items() if not name.startswith("wavlm.")
        }
    torch.save(
        {
            "model": model_state,
            "wavlm_omitted_because_frozen": model.frozen,
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "batch_in_epoch": batch_in_epoch,
            "running_loss": running_loss,
            "global_step": global_step,
            "speaker_to_index": speaker_to_index,
            "settings": settings,
        },
        temporary,
    )
    temporary.replace(path)


def _resume_position(state: dict[str, Any]) -> tuple[int, int, float]:
    """兼容旧 checkpoint，并返回 epoch、已完成 batch 数和累计 loss。"""
    epoch = int(state["epoch"])
    if bool(state.get("epoch_complete", True)):
        return epoch + 1, 0, 0.0
    return epoch, int(state.get("batch_in_epoch", 0)), float(state.get("running_loss", 0.0))


def _optimizer_step(
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
    accumulation: int,
    accumulated_batches: int,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """完成一次 optimizer step，并校正 epoch 尾部不足 accumulation 的梯度。"""
    if not 1 <= accumulated_batches <= accumulation:
        raise ValueError("accumulated_batches 必须位于 [1, accumulation]")
    if accumulated_batches != accumulation:
        scale = accumulation / accumulated_batches
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _epoch_indices(length: int, seed: int, epoch: int, shuffle: bool) -> list[int] | range:
    if not shuffle:
        return range(length)
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
    return torch.randperm(length, generator=generator).tolist()


def _forward_embeddings(
    model: WavLMECAPA,
    waveforms: torch.Tensor,
    device: torch.device,
    feature_microbatch_size: int | None,
) -> torch.Tensor:
    """冻结 WavLM 分块前向，再用完整物理 batch 训练 ECAPA/BatchNorm。"""
    batch_size = int(waveforms.shape[0])
    if not feature_microbatch_size or feature_microbatch_size >= batch_size:
        return model(waveforms.to(device, non_blocking=True))
    if not model.frozen:
        raise ValueError("feature_microbatch_size 仅支持冻结的 WavLM frontend")
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in waveforms.split(feature_microbatch_size):
            chunk_features, frame_mask = model.extract_features(
                chunk.to(device, non_blocking=True)
            )
            if frame_mask is not None:
                raise ValueError("分块 WavLM 当前要求固定长度、无 attention mask")
            features.append(chunk_features)
    return model.backend(torch.cat(features, dim=0))


def train_audio(
    config: dict[str, Any],
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    sph2pipe: str | None = None,
    resume: str | Path | None = None,
    init_from: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    if resume and init_from:
        raise ValueError("--resume 与 --init-from 不能同时使用")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_settings = config["train"]
    model_settings = config["model"]
    if model_settings.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = str(model_settings["hf_endpoint"])
    dataset = FisherTrainingDataset(
        manifest_path,
        split_path,
        "train",
        int(config["sample_rate"]),
        float(config["crop_seconds"]),
        seed,
        sph2pipe,
        train_settings.get("segment_cache_dir"),
        str(train_settings.get("sampling_mode", "one_per_call_side")),
        str(train_settings.get("short_utterance_mode", "zero_pad")),
    )
    batch_size = int(train_settings["batch_size"])
    batches_in_epoch = len(dataset) // batch_size
    if batches_in_epoch == 0:
        raise ValueError("训练数据不足一个完整 batch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WavLMECAPA.from_pretrained(
        model_settings["wavlm_name"],
        int(model_settings["embedding_dim"]),
        bool(model_settings["wavlm_frozen"]),
    ).to(device)
    classifier = AAMSoftmaxLoss(
        int(model_settings["embedding_dim"]),
        len(dataset.speaker_to_index),
        float(train_settings["aam_margin"]),
        float(train_settings["aam_scale"]),
    ).to(device)
    if init_from:
        initial = torch.load(init_from, map_location="cpu", weights_only=False)
        if initial.get("speaker_to_index") != dataset.speaker_to_index:
            raise ValueError("semi-informed 初始化要求 anonymized manifest 与 lazy checkpoint 使用同一 speaker 标签")
        model.load_state_dict(
            initial["model"], strict=not initial.get("wavlm_omitted_because_frozen", False)
        )
        classifier.load_state_dict(initial["classifier"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters.extend(classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_settings["learning_rate"]),
        weight_decay=float(train_settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_settings["scheduler_step"]),
        gamma=float(train_settings["scheduler_gamma"]),
    )
    use_amp = bool(train_settings.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    feature_microbatch_size = int(train_settings.get("feature_microbatch_size", 0)) or None
    if feature_microbatch_size is not None and feature_microbatch_size < 1:
        raise ValueError("feature_microbatch_size 必须为正整数")
    start_epoch = 0
    resume_batch = 0
    resumed_running_loss = 0.0
    global_step = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(
            state["model"], strict=not state.get("wavlm_omitted_because_frozen", False)
        )
        classifier.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch, resume_batch, resumed_running_loss = _resume_position(state)
        global_step = int(state["global_step"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "train.jsonl"
    accumulation = int(train_settings["gradient_accumulation"])
    checkpoint_interval = int(train_settings.get("checkpoint_interval_steps", 10))
    num_workers = int(train_settings.get("num_workers", 0))
    prefetch_factor = int(train_settings.get("prefetch_factor", 2))
    pin_memory = bool(train_settings.get("pin_memory", device.type == "cuda"))
    shuffle_utterances = bool(train_settings.get("shuffle_utterances", False))
    save_epoch_checkpoints = bool(train_settings.get("save_epoch_checkpoints", False))
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers 必须非负，prefetch_factor 必须为正整数")
    if accumulation < 1 or checkpoint_interval < 1:
        raise ValueError("gradient_accumulation 和 checkpoint_interval_steps 必须为正整数")
    optimizer.zero_grad(set_to_none=True)
    stopped_early = False
    last_checkpoint = output_path / "last.pt"
    for epoch in range(start_epoch, int(train_settings["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        classifier.train()
        first_batch = resume_batch if epoch == start_epoch else 0
        if first_batch < 0 or first_batch > batches_in_epoch:
            raise ValueError(f"checkpoint batch_in_epoch 越界: {first_batch}/{batches_in_epoch}")
        first_item = first_batch * batch_size
        epoch_order = _epoch_indices(len(dataset), seed, epoch, shuffle_utterances)
        epoch_dataset = Subset(dataset, epoch_order[first_item:])
        loader_kwargs: dict[str, Any] = {
            "dataset": epoch_dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": True,
        }
        if num_workers:
            loader_kwargs.update(
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
            )
        loader = DataLoader(**loader_kwargs)
        progress = tqdm(loader, desc=f"epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        running_loss = resumed_running_loss if epoch == start_epoch else 0.0
        epoch_started = time.time()
        accumulated_batches = 0
        batch_index = first_batch
        for batch_index, (waveforms, labels) in enumerate(progress, start=first_batch + 1):
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                embeddings = _forward_embeddings(
                    model, waveforms, device, feature_microbatch_size
                )
                loss = classifier(embeddings, labels) / accumulation
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            accumulated_batches += 1
            running_loss += float(loss.item()) * accumulation
            if accumulated_batches == accumulation:
                _optimizer_step(
                    parameters,
                    optimizer,
                    float(train_settings["gradient_clip"]),
                    accumulation,
                    accumulated_batches,
                    scaler,
                )
                accumulated_batches = 0
                global_step += 1
                progress.set_postfix(loss=f"{running_loss / batch_index:.4f}", step=global_step)
                if global_step % checkpoint_interval == 0:
                    _save_checkpoint(
                        last_checkpoint,
                        model,
                        classifier,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        dataset.speaker_to_index,
                        {"config": config, "manifest": str(manifest_path), "split": str(split_path)},
                        epoch_complete=False,
                        batch_in_epoch=batch_index,
                        running_loss=running_loss,
                        scaler=scaler,
                    )
                if max_steps is not None and global_step >= max_steps:
                    stopped_early = True
                    break
        if not stopped_early and accumulated_batches:
            _optimizer_step(
                parameters,
                optimizer,
                float(train_settings["gradient_clip"]),
                accumulation,
                accumulated_batches,
                scaler,
            )
            global_step += 1
            accumulated_batches = 0
        epoch_complete = not stopped_early and batch_index == batches_in_epoch
        if epoch_complete:
            scheduler.step()
        record = {
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "batch_in_epoch": batch_index,
            "batches_in_epoch": batches_in_epoch,
            "global_step": global_step,
            "loss": running_loss / max(1, batch_index),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            "checkpoint": str(last_checkpoint.resolve()),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _save_checkpoint(
            last_checkpoint,
            model,
            classifier,
            optimizer,
            scheduler,
            epoch,
            global_step,
            dataset.speaker_to_index,
            {"config": config, "manifest": str(manifest_path), "split": str(split_path)},
            epoch_complete=epoch_complete,
            batch_in_epoch=batch_index,
            running_loss=running_loss,
            scaler=scaler,
        )
        if epoch_complete and save_epoch_checkpoints:
            _save_checkpoint(
                output_path / f"epoch_{epoch:02d}.pt",
                model,
                classifier,
                optimizer,
                scheduler,
                epoch,
                global_step,
                dataset.speaker_to_index,
                {"config": config, "manifest": str(manifest_path), "split": str(split_path)},
                epoch_complete=True,
                batch_in_epoch=batch_index,
                running_loss=running_loss,
                scaler=scaler,
            )
        if stopped_early:
            break
    return {
        "checkpoint": str(last_checkpoint.resolve()),
        "log": str(log_path.resolve()),
        "global_step": global_step,
        "stopped_early": stopped_early,
        "resume_command": f"mmsv train-audio --config <config> --manifest {manifest_path} "
        f"--splits {split_path} --output-dir {output_dir} --resume {last_checkpoint}",
        "initialized_from": None if init_from is None else str(Path(init_from).resolve()),
    }


@torch.inference_mode()
def extract_embeddings(
    checkpoint: str | Path,
    manifest_path: str | Path,
    output_npz: str | Path,
    sample_rate: int,
    sph2pipe: str | None = None,
    limit: int | None = None,
    trial_jsonl: str | Path | None = None,
) -> dict[str, object]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = state["settings"]["config"]
    model_settings = settings["model"]
    if model_settings.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = str(model_settings["hf_endpoint"])
    model = WavLMECAPA.from_pretrained(
        model_settings["wavlm_name"],
        int(model_settings["embedding_dim"]),
        bool(model_settings["wavlm_frozen"]),
    )
    model.load_state_dict(
        state["model"], strict=not state.get("wavlm_omitted_because_frozen", False)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = read_manifest(manifest_path)
    requested_ids: set[str] | None = None
    if trial_jsonl is not None:
        requested_ids = set()
        with Path(trial_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                trial = json.loads(line)
                requested_ids.update(map(str, trial["enroll_utt_ids"]))
                requested_ids.update(map(str, trial["target_utt_ids"]))
        rows = [row for row in rows if row["utt_id"] in requested_ids]
        found_ids = {row["utt_id"] for row in rows}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            example = min(missing_ids)
            raise KeyError(f"trial 引用了 manifest 中不存在的 {len(missing_ids)} 条 utterance，例如 {example}")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("没有需要提取的 utterance")
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for row in tqdm(rows, desc="extract", unit="utt", dynamic_ncols=True):
        waveform = read_segment(row, sample_rate, sph2pipe)
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
        vector = model(tensor)[0].cpu().numpy().astype(np.float32)
        ids.append(row["utt_id"])
        vectors.append(vector)
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, utt_ids=np.asarray(ids), embeddings=np.stack(vectors))
    return {
        "output": str(output_path.resolve()),
        "utterances": len(ids),
        "trial_filter": None if trial_jsonl is None else str(Path(trial_jsonl).resolve()),
        "requested_utterances": None if requested_ids is None else len(requested_ids),
    }
