import torch

from mmsv.train import _optimizer_step, _resume_position


def test_resume_position_supports_legacy_and_mid_epoch_checkpoints():
    assert _resume_position({"epoch": 2}) == (3, 0, 0.0)
    assert _resume_position({
        "epoch": 2,
        "epoch_complete": False,
        "batch_in_epoch": 320,
        "running_loss": 123.5,
    }) == (2, 320, 123.5)


def test_partial_accumulation_is_rescaled_before_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.25])
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    _optimizer_step([parameter], optimizer, 100.0, accumulation=4, accumulated_batches=1)
    assert torch.allclose(parameter.detach(), torch.tensor([0.0]))
    assert parameter.grad is None
