from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def audit_protocol(
    trials_dir: Path,
    plan_path: Path,
    mapping_path: Path,
    original_embeddings_path: Path,
) -> dict[str, object]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapped_sessions = mapping["sessions"]

    plan_ids: set[str] = set()
    output_paths: set[str] = set()
    session_references: dict[str, set[tuple[str, str]]] = defaultdict(set)
    session_speakers: dict[str, set[str]] = defaultdict(set)
    minimum_reference_duration = float("inf")
    plan_rows = 0
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            plan_rows += 1
            utterance_id = row["utt_id"]
            session_id = row["session_id"]
            expected_session = f"{row['call_id']}:{row['channel']}"
            if session_id != expected_session:
                raise ValueError(
                    f"session key mismatch for {utterance_id}: {session_id} != {expected_session}"
                )
            if utterance_id in plan_ids:
                raise ValueError(f"duplicate plan utterance: {utterance_id}")
            plan_ids.add(utterance_id)
            output_path = row["output_audio_path"].replace("\\", "/").casefold()
            if output_path in output_paths:
                raise ValueError(f"duplicate output path: {row['output_audio_path']}")
            output_paths.add(output_path)
            session_references[session_id].add(
                (row["reference_utt_id"], row["reference_speaker_id"])
            )
            session_speakers[session_id].add(row["speaker_id"])
            minimum_reference_duration = min(
                minimum_reference_duration, float(row["reference_duration"])
            )

    inconsistent_references = sorted(
        session for session, references in session_references.items() if len(references) != 1
    )
    mixed_speaker_sessions = sorted(
        session for session, speakers in session_speakers.items() if len(speakers) != 1
    )
    if inconsistent_references:
        raise ValueError(f"sessions with multiple references: {inconsistent_references[:5]}")
    if mixed_speaker_sessions:
        raise ValueError(f"sessions with multiple speakers: {mixed_speaker_sessions[:5]}")
    if minimum_reference_duration <= 4.0:
        raise ValueError(
            f"reference duration must be >4 seconds, found {minimum_reference_duration}"
        )
    if set(mapped_sessions) != set(session_references):
        raise ValueError("pseudo mapping session set does not equal plan session set")
    for session_id, references in session_references.items():
        reference_utt_id, reference_speaker_id = next(iter(references))
        mapped = mapped_sessions[session_id]
        if (
            mapped["reference_utt_id"] != reference_utt_id
            or mapped["reference_speaker_id"] != reference_speaker_id
        ):
            raise ValueError(f"mapping/plan reference mismatch for {session_id}")

    archive = np.load(original_embeddings_path, allow_pickle=False)
    embedding_ids = set(archive["utt_ids"].astype(str).tolist())
    target_ids: set[str] = set()
    enrollment_ids: set[str] = set()
    trial_ids: set[str] = set()
    per_seed: dict[int, dict[str, int]] = {}
    for seed in [1, 2, 3, 4, 5]:
        path = trials_dir / f"sampling_seed_{seed}.jsonl"
        counts = {"target": 0, "nontarget": 0, "records": 0}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                trial = json.loads(line)
                if int(trial["seed"]) != seed:
                    raise ValueError(f"wrong seed in {path}: {trial['seed']}")
                trial_id = str(trial["trial_id"])
                if trial_id in trial_ids:
                    raise ValueError(f"duplicate trial id: {trial_id}")
                trial_ids.add(trial_id)
                enroll = list(map(str, trial["enroll_utt_ids"]))
                target = list(map(str, trial["target_utt_ids"]))
                if len(enroll) != 15 or len(set(enroll)) != 15:
                    raise ValueError(f"invalid enrollment list in {trial_id}")
                if len(target) != 15 or len(set(target)) != 15:
                    raise ValueError(f"invalid target list in {trial_id}")
                if set(enroll) & set(target):
                    raise ValueError(f"enrollment/target utterance overlap in {trial_id}")
                enroll_call = str(trial["enroll_session_id"]).rsplit(":", 1)[0]
                target_call = str(trial["target_session_id"]).rsplit(":", 1)[0]
                if enroll_call == target_call:
                    raise ValueError(f"call leakage in {trial_id}")
                enrollment_ids.update(enroll)
                target_ids.update(target)
                label = int(trial["label"])
                counts["target" if label == 1 else "nontarget"] += 1
                counts["records"] += 1
        per_seed[seed] = counts

    if plan_ids != target_ids:
        raise ValueError(
            f"plan/target union mismatch: plan_only={len(plan_ids-target_ids)}, "
            f"target_only={len(target_ids-plan_ids)}"
        )
    missing_embeddings = (target_ids | enrollment_ids) - embedding_ids
    if missing_embeddings:
        raise ValueError(
            f"original embeddings miss {len(missing_embeddings)} trial utterances, "
            f"e.g. {min(missing_embeddings)}"
        )

    reference_use: dict[tuple[str, str], int] = defaultdict(int)
    for references in session_references.values():
        reference_use[next(iter(references))] += 1
    collision_sessions = sum(count - 1 for count in reference_use.values() if count > 1)
    result = {
        "valid": True,
        "protocol": "fixed-original-enrollment/increasing-session-fixed-anonymous-target",
        "session_definition": "fisher_call_id_plus_channel",
        "plan_rows": plan_rows,
        "unique_plan_utterances": len(plan_ids),
        "unique_output_paths": len(output_paths),
        "target_union_utterances": len(target_ids),
        "enrollment_union_utterances": len(enrollment_ids),
        "trial_ids": len(trial_ids),
        "sessions": len(session_references),
        "unique_reference_utterances": len(reference_use),
        "reference_collision_sessions": collision_sessions,
        "minimum_reference_duration": minimum_reference_duration,
        "embedding_universe_utterances": len(embedding_ids),
        "missing_embedding_utterances": 0,
        "inconsistent_reference_sessions": 0,
        "mixed_speaker_sessions": 0,
        "per_seed": per_seed,
        "inputs": {
            "trials_dir": str(trials_dir.resolve()),
            "plan": str(plan_path.resolve()),
            "mapping": str(mapping_path.resolve()),
            "original_embeddings": str(original_embeddings_path.resolve()),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly audit the SAAR Phase 1/2 protocol")
    parser.add_argument("--trials-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--original-embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_protocol(
        args.trials_dir, args.plan, args.mapping, args.original_embeddings
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
