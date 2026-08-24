import torch

from mmsv.train import (
    FisherTrainingDataset,
    _epoch_indices,
    _forward_embeddings,
    _optimizer_step,
    _resume_position,
)


def test_segment_cache_path_is_stable_and_sharded(tmp_path):
    first = FisherTrainingDataset.cache_path(tmp_path, "fe_03_00001_A_0001")
    second = FisherTrainingDataset.cache_path(tmp_path, "fe_03_00001_A_0001")
    assert first == second
    assert first.parent.parent == tmp_path
    assert len(first.parent.name) == 2
    assert first.name == "fe_03_00001_A_0001.flac"


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


def test_epoch_indices_are_deterministic_shuffled_permutations():
    first = _epoch_indices(20, seed=1234, epoch=2, shuffle=True)
    second = _epoch_indices(20, seed=1234, epoch=2, shuffle=True)
    different_epoch = _epoch_indices(20, seed=1234, epoch=3, shuffle=True)
    assert first == second
    assert first != different_epoch
    assert sorted(first) == list(range(20))


def test_frozen_frontend_microbatches_feed_one_physical_backend_batch():
    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 2)
            self.seen_batch = 0

        def forward(self, features):
            self.seen_batch = features.shape[0]
            return self.projection(features.mean(dim=1))

    class FakeModel:
        frozen = True

        def __init__(self):
            self.backend = Backend()
            self.chunk_sizes = []

        def extract_features(self, waveforms, attention_mask=None):
            self.chunk_sizes.append(waveforms.shape[0])
            return waveforms.unsqueeze(-1), None

    model = FakeModel()
    output = _forward_embeddings(
        model, torch.arange(18, dtype=torch.float32).reshape(6, 3), torch.device("cpu"), 2
    )
    output.sum().backward()
    assert model.chunk_sizes == [2, 2, 2]
    assert model.backend.seen_batch == 6
    assert model.backend.projection.weight.grad is not None
