from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _length_mask(lengths: torch.Tensor | None, max_length: int, device: torch.device) -> torch.Tensor:
    if lengths is None:
        return torch.ones((1, max_length), dtype=torch.bool, device=device)
    if lengths.dtype.is_floating_point and torch.all(lengths <= 1.0):
        lengths = torch.round(lengths * max_length).long()
    else:
        lengths = lengths.long()
    return torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)


class TDNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Res2NetBlock(nn.Module):
    def __init__(self, channels: int, scale: int, kernel_size: int, dilation: int):
        super().__init__()
        if channels % scale:
            raise ValueError("channels 必须能被 Res2Net scale 整除")
        width = channels // scale
        self.scale = scale
        self.blocks = nn.ModuleList(
            TDNNBlock(width, width, kernel_size, dilation) for _ in range(scale - 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(inputs, self.scale, dim=1)
        outputs = [chunks[0]]
        running = None
        for index, block in enumerate(self.blocks):
            running = chunks[index + 1] if running is None else chunks[index + 1] + running
            running = block(running)
            outputs.append(running)
        return torch.cat(outputs, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            summary = inputs.mean(dim=-1, keepdim=True)
        else:
            weights = mask.to(inputs.dtype).unsqueeze(1)
            summary = (inputs * weights).sum(dim=-1, keepdim=True) / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
        return inputs * self.projection(summary)


class SERes2NetBlock(nn.Module):
    def __init__(self, channels: int, scale: int, kernel_size: int, dilation: int, se_channels: int):
        super().__init__()
        self.pre = TDNNBlock(channels, channels, 1)
        self.res2net = Res2NetBlock(channels, scale, kernel_size, dilation)
        self.post = TDNNBlock(channels, channels, 1)
        self.se = SEBlock(channels, se_channels)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.post(self.res2net(self.pre(inputs)))
        return inputs + self.se(output, mask)


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128, global_context: bool = True):
        super().__init__()
        self.global_context = global_context
        attention_input = channels * 3 if global_context else channels
        self.attention = nn.Sequential(
            nn.Conv1d(attention_input, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, time = inputs.shape
        if mask is None:
            mask = torch.ones((batch, time), dtype=torch.bool, device=inputs.device)
        weights = mask.to(inputs.dtype).unsqueeze(1)
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (inputs * weights).sum(dim=-1, keepdim=True) / denominator
        variance = ((inputs - mean).square() * weights).sum(dim=-1, keepdim=True) / denominator
        std = variance.clamp_min(1.0e-5).sqrt()
        attention_input = (
            torch.cat([inputs, mean.expand_as(inputs), std.expand_as(inputs)], dim=1)
            if self.global_context else inputs
        )
        logits = self.attention(attention_input).masked_fill(~mask.unsqueeze(1), -torch.inf)
        alpha = torch.softmax(logits, dim=-1)
        pooled_mean = (alpha * inputs).sum(dim=-1)
        pooled_second = (alpha * inputs.square()).sum(dim=-1)
        pooled_std = (pooled_second - pooled_mean.square()).clamp_min(1.0e-5).sqrt()
        return torch.cat([pooled_mean, pooled_std], dim=1)


class ECAPABackend(nn.Module):
    """ECAPA-TDNN backend with an explicit pre-ASP interface for frame concatenation."""

    def __init__(
        self,
        input_dim: int = 1024,
        channels: int = 512,
        mfa_channels: int = 1536,
        embedding_dim: int = 192,
        res2net_scale: int = 8,
    ):
        super().__init__()
        self.initial = TDNNBlock(input_dim, channels, 5)
        self.blocks = nn.ModuleList(
            SERes2NetBlock(channels, res2net_scale, 3, dilation, 128)
            for dilation in (2, 3, 4)
        )
        self.mfa = TDNNBlock(channels * 3, mfa_channels, 1)
        self.asp = AttentiveStatisticsPooling(mfa_channels)
        self.asp_bn = nn.BatchNorm1d(mfa_channels * 2)
        self.projection = nn.Linear(mfa_channels * 2, embedding_dim)
        self.frame_dim = mfa_channels
        self.embedding_dim = embedding_dim

    def extract_pre_asp_frames(
        self, features: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 输入 [B,T,C]，输出 [B,T,C_mfa]。
        output = self.initial(features.transpose(1, 2))
        collected = []
        for block in self.blocks:
            output = block(output, frame_mask)
            collected.append(output)
        return self.mfa(torch.cat(collected, dim=1)).transpose(1, 2)

    def pool_and_project(
        self, frames: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = self.asp(frames.transpose(1, 2), frame_mask)
        embedding = self.projection(self.asp_bn(pooled))
        return F.normalize(embedding, dim=-1)

    def forward(
        self, features: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.pool_and_project(self.extract_pre_asp_frames(features, frame_mask), frame_mask)


class WavLMECAPA(nn.Module):
    def __init__(self, wavlm: nn.Module, input_dim: int, embedding_dim: int = 192, frozen: bool = True):
        super().__init__()
        self.wavlm = wavlm
        self.backend = ECAPABackend(input_dim=input_dim, embedding_dim=embedding_dim)
        self.frozen = frozen
        if frozen:
            self.wavlm.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls, model_name: str = "microsoft/wavlm-large", embedding_dim: int = 192, frozen: bool = True
    ) -> "WavLMECAPA":
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from transformers import WavLMModel

        model_path = Path(model_name)
        if not model_path.exists():
            model_path = Path(snapshot_download(
                repo_id=model_name,
                allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"],
            ))
        # 从本地 snapshot 加载，避免 Transformers 启动非 daemon 的 safetensors 转换线程。
        wavlm = WavLMModel.from_pretrained(
            model_path, local_files_only=True, use_safetensors=False
        )
        return cls(wavlm, int(wavlm.config.hidden_size), embedding_dim, frozen)

    def train(self, mode: bool = True) -> "WavLMECAPA":
        super().train(mode)
        if self.frozen:
            self.wavlm.eval()  # 冻结前端时同时关闭 dropout，保证特征可重复。
        return self

    def extract_features(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            output = self.wavlm(input_values=waveforms, attention_mask=attention_mask)
        features = output.last_hidden_state
        frame_mask = None
        if attention_mask is not None and hasattr(self.wavlm, "_get_feature_vector_attention_mask"):
            frame_mask = self.wavlm._get_feature_vector_attention_mask(features.shape[1], attention_mask)
        return features, frame_mask

    def extract_pre_asp_frames(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features, frame_mask = self.extract_features(waveforms, attention_mask)
        return self.backend.extract_pre_asp_frames(features, frame_mask), frame_mask

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        features, frame_mask = self.extract_features(waveforms, attention_mask)
        return self.backend(features, frame_mask)


class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, labels)


@dataclass
class TinyWavLMOutput:
    last_hidden_state: torch.Tensor
