import csv
import json
from pathlib import Path

import numpy as np

from mmsv.anonymization import build_anonymization_plan
from mmsv.data.session_trials import build_session_trial_sets, validate_session_trials
from mmsv.metrics import compute_pcs, score_session_trials, summarize_privacy_curve


def _write_fisher_fixture(root: Path) -> tuple[Path, Path]:
    manifest = root / "manifest.csv"
    fields = [
        "utt_id", "speaker_id", "call_id", "channel", "audio_path",
        "start", "end", "duration", "transcript",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for speaker_index, speaker_id in enumerate(["s1", "s2"]):
            for call_index in range(3):
                call_id = f"c{speaker_index}-{call_index}"
                for utterance_index in range(20):
                    writer.writerow({
                        "utt_id": f"{speaker_id}-{call_index}-{utterance_index:02d}",
                        "speaker_id": speaker_id,
                        "call_id": call_id,
                        "channel": str(speaker_index),
                        "audio_path": f"{call_id}.sph",
                        "start": str(utterance_index),
                        "end": str(utterance_index + 1),
                        "duration": "1",
                        "transcript": "fixture",
                    })
    splits = root / "splits.csv"
    with splits.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker_id", "split"])
        writer.writeheader()
        writer.writerows([
            {"speaker_id": "s1", "split": "evaluation"},
            {"speaker_id": "s2", "split": "evaluation"},
        ])
    return manifest, splits


def test_session_trials_are_nested_fixed_and_reproducible(tmp_path: Path) -> None:
    manifest, splits = _write_fisher_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = dict(
        manifest_path=manifest,
        split_path=splits,
        split_name="evaluation",
        n_values=[1, 2, 5, 10, 15],
        enrollment_n=15,
        seeds=[1, 2],
    )
    audit = build_session_trial_sets(output_dir=first, **kwargs)
    build_session_trial_sets(output_dir=second, **kwargs)
    assert audit["eligible_speakers"] == 2
    for seed in [1, 2]:
        left = first / f"sampling_seed_{seed}.jsonl"
        right = second / f"sampling_seed_{seed}.jsonl"
        assert left.read_bytes() == right.read_bytes()
        assert validate_session_trials(left, manifest, [1, 2, 5, 10, 15], 15) == {
            "target": 2,
            "nontarget": 2,
        }
        trial = json.loads(left.read_text(encoding="utf-8").splitlines()[0])
        assert len(trial["enroll_utt_ids"]) == 15
        assert trial["target_utt_ids"][:2][:1] == trial["target_utt_ids"][:1]


def test_session_pseudo_mapping_is_persisted_and_target_only(tmp_path: Path) -> None:
    manifest, splits = _write_fisher_fixture(tmp_path)
    trials = tmp_path / "trials"
    build_session_trial_sets(
        manifest, splits, trials, "evaluation", [1, 2, 5, 10, 15], 15, [1]
    )
    pool = tmp_path / "pool.csv"
    with pool.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utt_id", "speaker_id", "audio_path", "duration", "subset"],
        )
        writer.writeheader()
        for index in range(20):
            writer.writerow({
                "utt_id": f"ref-{index}",
                "speaker_id": f"ref-speaker-{index}",
                "audio_path": f"ref-{index}.flac",
                "duration": "5",
                "subset": "train-clean-360",
            })

    mapping_path = tmp_path / "mapping.json"
    plan_path = tmp_path / "plan.csv"
    audit = build_anonymization_plan(
        manifest,
        pool,
        plan_path,
        tmp_path / "audio",
        2027,
        trials / "all_seeds.jsonl",
        reference_mapping="session",
        mapping_output_json=mapping_path,
        trial_role="target",
    )
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_session: dict[str, set[str]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], set()).add(row["reference_utt_id"])
    assert all(len(references) == 1 for references in by_session.values())
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert set(mapping["sessions"]) == set(by_session)
    assert len({entry["pseudo_seed"] for entry in mapping["sessions"].values()}) == len(by_session)
    assert audit["trial_role"] == "target"
    assert audit["mapping"] == "per_session_deterministic_random"


def test_session_scoring_keeps_enrollment_fixed_and_computes_pcs(tmp_path: Path) -> None:
    trials = tmp_path / "seed_1.jsonl"
    trials.write_text(
        "\n".join([
            json.dumps({
                "trial_id": "tar", "seed": 1, "label": 1,
                "enroll_session_id": "e", "target_session_id": "t",
                "enroll_utt_ids": ["e1", "e2"], "target_utt_ids": ["t1", "t2"],
            }),
            json.dumps({
                "trial_id": "non", "seed": 1, "label": 0,
                "enroll_session_id": "e", "target_session_id": "i",
                "enroll_utt_ids": ["e1", "e2"], "target_utt_ids": ["i1", "i2"],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    original = {
        "e1": np.array([1.0, 0.0]),
        "e2": np.array([1.0, 0.0]),
    }
    anonymized = {
        "t1": np.array([1.0, 0.0]),
        "t2": np.array([0.8, 0.2]),
        "i1": np.array([0.0, 1.0]),
        "i2": np.array([0.2, 0.8]),
    }
    result = score_session_trials(
        trials, original, anonymized, "O-A", 1, tmp_path / "scores.csv"
    )
    assert result["enrollment_n"] == 2
    assert result["eer"] == 0.0
    pcs = compute_pcs([trials], anonymized, tmp_path / "pcs.csv")
    assert -1.0 <= pcs["pcs"] <= 1.0


def test_privacy_summary_reports_delta_and_slope(tmp_path: Path) -> None:
    paths = []
    for n, eer in [(1, 0.4), (2, 0.35), (5, 0.3)]:
        path = tmp_path / f"n{n}.json"
        path.write_text(json.dumps({
            "seed": 1, "n": n, "eer": eer, "score_file": f"n{n}.csv",
        }), encoding="utf-8")
        paths.append(path)
    result = summarize_privacy_curve(
        paths,
        tmp_path / "summary.csv",
        system="session-fixed",
        attacker="mean",
        checkpoint="checkpoint.pt",
        git_commit="abc",
    )
    assert result["slope_beta_mean"] < 0.0
    with (tmp_path / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    n5 = next(row for row in rows if row["N"] == "5")
    assert np.isclose(float(n5["delta_eer"]), 0.1)
    assert np.isclose(float(n5["relative_degradation"]), 25.0)
    assert Path(n5["metrics_file"]).name == "n5.json"
