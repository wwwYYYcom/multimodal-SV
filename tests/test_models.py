import torch

from mmsv.aggregation import QueryAttention, concatenate_frames, mean_pool
from mmsv.models import ECAPABackend
from mmsv.multimodal import AudioProsodyFusion, AudioTextFusion, summarize_prosody, weighted_fusion


def test_aggregators_and_ecapa_shapes() -> None:
    torch.manual_seed(1)
    utterances = torch.randn(2, 5, 24)
    assert mean_pool(utterances).shape == (2, 24)
    assert QueryAttention(24, 4, 0.3)(utterances).shape == (2, 24)
    assert concatenate_frames([torch.randn(3, 16), torch.randn(4, 16)]).shape == (7, 16)

    backend = ECAPABackend(input_dim=16, channels=32, mfa_channels=64, embedding_dim=24)
    features = torch.randn(2, 40, 16)
    mask = torch.ones(2, 40, dtype=torch.bool)
    frames = backend.extract_pre_asp_frames(features, mask)
    embeddings = backend.pool_and_project(frames, mask)
    assert frames.shape == (2, 40, 64)
    assert embeddings.shape == (2, 24)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2), atol=1e-5)

    prosody = summarize_prosody(torch.randn(2, 5, 3))
    assert prosody.shape == (2, 6)
    audio = torch.randn(2, 192)
    text = torch.randn(2, 512)
    assert AudioTextFusion()(audio, text).shape == (2, 192)
    assert AudioProsodyFusion()(audio, torch.randn(2, 3)).shape == (2, 192)
    assert weighted_fusion(audio, torch.randn(2, 192), 0.5).shape == (2, 192)
