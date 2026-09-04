from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def freeze_privacy_encoder(encoder: nn.Module) -> nn.Module:
    """Freeze encoder weights without disabling gradients to its waveform input."""

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def nested_aggregates(
    embeddings: torch.Tensor,
    n_values: Sequence[int] = (1, 2, 4),
) -> dict[int, torch.Tensor]:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, utterance, dimension]")
    ordered_n = sorted(set(int(value) for value in n_values))
    if not ordered_n or ordered_n[0] <= 0 or ordered_n[-1] > embeddings.shape[1]:
        raise ValueError("n_values must be positive and no larger than K")
    normalized = F.normalize(embeddings, dim=-1)
    return {
        n: F.normalize(normalized[:, :n].mean(dim=1), dim=-1)
        for n in ordered_n
    }


def source_leakage_loss(
    aggregates: Sequence[torch.Tensor],
    source_center: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if not aggregates:
        raise ValueError("at least one aggregate is required")
    source_center = F.normalize(source_center, dim=-1)
    similarities = [
        F.cosine_similarity(aggregate, source_center, dim=-1)
        for aggregate in aggregates
    ]
    loss = torch.stack([
        F.relu(similarity - float(margin)).square().mean()
        for similarity in similarities
    ]).mean()
    return loss, similarities


def cumulative_growth_loss(
    similarities: Sequence[torch.Tensor],
    delta: float = 0.0,
) -> torch.Tensor:
    if len(similarities) < 2:
        raise ValueError("growth loss requires at least two aggregation sizes")
    return torch.stack([
        F.relu(larger - smaller - float(delta)).mean()
        for smaller, larger in zip(similarities, similarities[1:])
    ]).sum()


def pseudo_anchor_loss(
    anonymous_embeddings: torch.Tensor,
    aggregates: Sequence[torch.Tensor],
    pseudo_center: torch.Tensor,
    aggregate_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not aggregates:
        raise ValueError("at least one aggregate is required")
    anonymous_embeddings = F.normalize(anonymous_embeddings, dim=-1)
    pseudo_center = F.normalize(pseudo_center, dim=-1)
    utterance_similarity = F.cosine_similarity(
        anonymous_embeddings,
        pseudo_center[:, None, :],
        dim=-1,
    )
    aggregate_similarity = torch.stack([
        F.cosine_similarity(aggregate, pseudo_center, dim=-1)
        for aggregate in aggregates
    ])
    utterance_loss = (1.0 - utterance_similarity).mean()
    aggregate_loss = (1.0 - aggregate_similarity).mean()
    return (
        utterance_loss + float(aggregate_weight) * aggregate_loss,
        utterance_similarity.mean(),
        aggregate_similarity.mean(),
    )


def saar_privacy_losses(
    anonymous_embeddings: torch.Tensor,
    source_center: torch.Tensor,
    pseudo_center: torch.Tensor,
    *,
    n_values: Sequence[int] = (1, 2, 4),
    source_margin: float = 0.2,
    growth_delta: float = 0.0,
    pseudo_aggregate_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    aggregate_by_n = nested_aggregates(anonymous_embeddings, n_values)
    ordered_n = sorted(aggregate_by_n)
    aggregates = [aggregate_by_n[n] for n in ordered_n]
    source_loss, similarities = source_leakage_loss(
        aggregates,
        source_center,
        source_margin,
    )
    growth_loss = cumulative_growth_loss(similarities, growth_delta)
    pseudo_loss, pseudo_sim_utt, pseudo_sim_agg = pseudo_anchor_loss(
        anonymous_embeddings,
        aggregates,
        pseudo_center,
        pseudo_aggregate_weight,
    )
    output: dict[str, torch.Tensor] = {
        "loss_src": source_loss,
        "loss_growth": growth_loss,
        "loss_pseudo": pseudo_loss,
        "pseudo_sim_utt": pseudo_sim_utt,
        "pseudo_sim_agg": pseudo_sim_agg,
    }
    output.update({
        f"source_similarity_n{n}": similarity.mean()
        for n, similarity in zip(ordered_n, similarities)
    })
    return output
