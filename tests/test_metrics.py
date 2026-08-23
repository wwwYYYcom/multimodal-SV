import numpy as np

from mmsv.metrics import compute_eer


def test_eer_perfect_and_random_like() -> None:
    eer, _ = compute_eer(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert eer == 0.0
    eer, _ = compute_eer(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.1, 0.0]))
    assert 0.0 <= eer <= 1.0

