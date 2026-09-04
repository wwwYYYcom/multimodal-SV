import torch
from torch import nn

from mmsv.data.session_batch import SessionBatchSampler
from mmsv.saar_losses import (
    cumulative_growth_loss,
    freeze_privacy_encoder,
    nested_aggregates,
    saar_privacy_losses,
)


def test_nested_aggregates_are_normalized() -> None:
    output = nested_aggregates(torch.randn(3, 4, 8), [1, 2, 4])
    assert list(output) == [1, 2, 4]
    for value in output.values():
        assert torch.allclose(value.norm(dim=-1), torch.ones(3), atol=1.0e-6)


def test_growth_loss_positive_and_zero() -> None:
    increasing = [torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3])]
    decreasing = [torch.tensor([0.3]), torch.tensor([0.2]), torch.tensor([0.1])]
    assert cumulative_growth_loss(increasing, 0.0).item() > 0.0
    assert cumulative_growth_loss(decreasing, 0.0).item() == 0.0


def test_frozen_encoder_keeps_input_gradient() -> None:
    anonymizer = nn.Linear(4, 4, bias=False)
    privacy_encoder = freeze_privacy_encoder(nn.Linear(4, 3, bias=False))
    waveform_features = torch.randn(2, 4)
    anonymous = anonymizer(waveform_features)
    embedding = privacy_encoder(anonymous)
    embedding.square().mean().backward()
    assert all(parameter.grad is None for parameter in privacy_encoder.parameters())
    assert anonymizer.weight.grad is not None
    assert torch.count_nonzero(anonymizer.weight.grad).item() > 0


def test_saar_loss_reaches_anonymizer() -> None:
    anonymizer = nn.Linear(6, 8)
    inputs = torch.randn(2, 4, 6)
    anonymous = anonymizer(inputs)
    losses = saar_privacy_losses(
        anonymous,
        torch.randn(2, 8),
        torch.randn(2, 8),
        n_values=[1, 2, 4],
        source_margin=-1.0,
    )
    total = losses["loss_src"] + losses["loss_growth"] + losses["loss_pseudo"]
    total.backward()
    assert anonymizer.weight.grad is not None
    assert torch.count_nonzero(anonymizer.weight.grad).item() > 0


def test_session_batch_sampler_is_reproducible_and_isolated() -> None:
    rows = []
    for call_id in ["c1", "c2"]:
        for index in range(5):
            rows.append({
                "speaker_id": "s1",
                "call_id": call_id,
                "channel": "0",
                "utt_id": f"{call_id}-{index}",
            })
    first = SessionBatchSampler(rows, 4, seed=9)
    second = SessionBatchSampler(rows, 4, seed=9)
    batches = list(first)
    assert batches == list(second)
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 4
        assert len({rows[index]["call_id"] for index in batch}) == 1
