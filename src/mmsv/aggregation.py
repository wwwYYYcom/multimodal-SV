from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def mean_pool(embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pooling followed by L2 normalization."""
    if mask is None:
        pooled = embeddings.mean(dim=-2)
    else:
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
    return F.normalize(pooled, dim=-1)


class QueryAttention(nn.Module):
    """论文 III-A1 的最简可审计实现；未添加论文未说明的 FFN/dropout。"""

    def __init__(self, dim: int = 192, heads: int = 4, temperature: float = 0.3):
        super().__init__()
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.query, mean=0.0, std=dim**-0.5)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        query = self.query.expand(batch_size, -1, -1) / self.temperature
        key_padding_mask = None if mask is None else ~mask.bool()
        pooled, _ = self.attention(
            query, embeddings, embeddings, key_padding_mask=key_padding_mask, need_weights=False
        )
        return F.normalize(pooled[:, 0], dim=-1)


def concatenate_frames(frame_sequences: Sequence[torch.Tensor]) -> torch.Tensor:
    if not frame_sequences:
        raise ValueError("frame_sequences 不能为空")
    feature_dim = frame_sequences[0].shape[-1]
    if any(sequence.ndim != 2 or sequence.shape[-1] != feature_dim for sequence in frame_sequences):
        raise ValueError("所有 frame sequence 必须是 [T, C] 且 C 相同")
    return torch.cat(list(frame_sequences), dim=0)

