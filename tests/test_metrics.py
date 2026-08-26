import numpy as np

from mmsv.metrics import compute_eer, score_mean_trials


def test_eer_perfect_and_random_like() -> None:
    eer, _ = compute_eer(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert eer == 0.0
    eer, _ = compute_eer(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.1, 0.0]))
    assert 0.0 <= eer <= 1.0


def test_score_mean_supports_single_utterance(tmp_path) -> None:
    trials = tmp_path / "trials.jsonl"
    trials.write_text(
        """{"trial_id":"target","label":1,"enroll_utt_ids":["a1","a2"],"target_utt_ids":["a3","a4"]}
{"trial_id":"nontarget","label":0,"enroll_utt_ids":["a1","a2"],"target_utt_ids":["b1","b2"]}
""",
        encoding="utf-8",
    )
    embeddings = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.9, 0.1], dtype=np.float32),
        "a3": np.array([1.0, 0.0], dtype=np.float32),
        "a4": np.array([0.8, 0.2], dtype=np.float32),
        "b1": np.array([0.0, 1.0], dtype=np.float32),
        "b2": np.array([0.1, 0.9], dtype=np.float32),
    }
    result = score_mean_trials(trials, embeddings, None, "O-O", 1, tmp_path / "n1.csv")
    assert result["n"] == 1
    assert result["trials"] == 2
    assert result["eer"] == 0.0
