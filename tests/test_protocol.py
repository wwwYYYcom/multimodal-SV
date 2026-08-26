import csv
import json
import random
from pathlib import Path

from mmsv.data.splits import split_speakers
from mmsv.data.trials import nested_sample, validate_trials


def test_nested_sample_is_prefix_nested() -> None:
    result = nested_sample([str(i) for i in range(30)], [1, 5, 10, 15], random.Random(3))
    assert result[1] == result[5][:1]
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
    assert validate_trials(path, [1, 5, 10, 15]) == {"target": 1, "nontarget": 0}


def test_split_reserves_trial_capable_validation_and_evaluation_speakers(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker_id", "call_id", "utt_id"])
        writer.writeheader()
        for speaker_index in range(4):
            for call_index in range(2):
                for utt_index in range(3):
                    writer.writerow({
                        "speaker_id": f"multi-{speaker_index}",
                        "call_id": f"m-{speaker_index}-{call_index}",
                        "utt_id": f"m-{speaker_index}-{call_index}-{utt_index}",
                    })
        for speaker_index in range(4):
            for utt_index in range(6):
                writer.writerow({
                    "speaker_id": f"single-{speaker_index}",
                    "call_id": f"s-{speaker_index}",
                    "utt_id": f"s-{speaker_index}-{utt_index}",
                })

    output = tmp_path / "splits.csv"
    audit = split_speakers(
        manifest,
        output,
        train_count=4,
        validation_count=2,
        evaluation_count=2,
        seed=7,
        require_exact_counts=True,
        evaluation_min_calls=2,
        validation_min_calls=2,
        trial_max_n=3,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    trial_splits = [row for row in rows if row["split"] in {"validation", "evaluation"}]
    assert len(trial_splits) == 4
    assert all(int(row["n_calls"]) >= 2 for row in trial_splits)
    assert audit["actual_counts"] == {"train": 4, "validation": 2, "evaluation": 2}
