import numpy as np
import pytest

from scripts.saar_robustness import cluster_weights, weighted_eer


def test_tied_scores_do_not_depend_on_trial_order():
    assert weighted_eer([1, 0], [.5, .5], [1, 1]) == .5
    assert weighted_eer([0, 1], [.5, .5], [1, 1]) == .5
    assert weighted_eer([0, 1], [0, 1], [2, 1]) == 0
    assert weighted_eer([0, 1], [1, 0], [2, 1]) == 1


def test_integer_weight_matches_replicated_threshold_roc():
    y = np.array([1, 0, 1, 0])
    s = np.array([.8, .7, .2, .2])
    w = np.array([3, 1, 2, 4])
    assert weighted_eer(y, s, w) == pytest.approx(
        weighted_eer(np.repeat(y, w), np.repeat(s, w), np.ones(w.sum())))


def test_shared_speaker_weights_preserve_genuine_cluster_and_both_impostor_roles():
    ts = [dict(enroll_speaker='a', target_speaker='a', label=1),
          dict(enroll_speaker='a', target_speaker='b', label=0),
          dict(enroll_speaker='b', target_speaker='a', label=0)]
    index = dict(a=0, b=1)
    np.testing.assert_array_equal(cluster_weights(ts, index, [2, 3], 'enrollment_speaker'), [2, 2, 3])
    np.testing.assert_array_equal(cluster_weights(ts, index, [2, 3], 'shared_speaker_dyadic'), [2, 6, 6])
    np.testing.assert_array_equal(cluster_weights(ts, index, [2, 0], 'shared_speaker_dyadic'), [2, 0, 0])


def test_empty_bootstrap_class_is_rejected():
    with pytest.raises(ValueError, match='empty class'):
        weighted_eer([1, 0], [.1, .2], [1, 0])
