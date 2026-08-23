import json
import random
from pathlib import Path

from mmsv.data.trials import nested_sample, validate_trials


def test_nested_sample_is_prefix_nested() -> None:
    result = nested_sample([str(i) for i in range(30)], [5, 10, 15], random.Random(3))
    assert result[5] == result[10][:5]
    assert result[10] == result[15][:10]


def test_validate_trials(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    trial = {
        "trial_id": "x",
        "label": 1,
        "enroll_utt_ids": [f"e{i}" for i in range(15)],
        "target_utt_ids": [f"t{i}" for i in range(15)],
    }
    path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
    assert validate_trials(path, [5, 10, 15]) == {"target": 1, "nontarget": 0}

