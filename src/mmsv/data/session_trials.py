from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .fisher import iter_manifest
from .trials import read_splits


def fisher_session_id(row: dict[str, str]) -> str:
    """Return a speaker-session key for a Fisher call side.

    A Fisher call contains two different speakers, so ``call_id`` alone is not
    a valid pseudo-speaker cache key.  The channel is part of the session key.
    """

    call_id = str(row["call_id"]).strip()
    channel = str(row["channel"]).strip()
    if not call_id or not channel:
        raise ValueError("Fisher session requires non-empty call_id and channel")
    return f"{call_id}:{channel}"


def _call_id(session_id: str) -> str:
    return session_id.rsplit(":", 1)[0]


def _load_session_pools(
    manifest_path: str | Path,
    split_path: str | Path,
    split_name: str,
    allowed_utterance_ids: set[str] | None = None,
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, tuple[str, str, str]],
]:
    split_for = read_splits(split_path)
    pools: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    utterance_meta: dict[str, tuple[str, str, str]] = {}
    for row in iter_manifest(manifest_path):
        if allowed_utterance_ids is not None and row["utt_id"] not in allowed_utterance_ids:
            continue
        speaker_id = row["speaker_id"]
        if split_for.get(speaker_id) != split_name:
            continue
        session_id = fisher_session_id(row)
        pools[speaker_id][session_id].append(row["utt_id"])
        utterance_meta[row["utt_id"]] = (speaker_id, session_id, row["call_id"])
    normalized = {
        speaker_id: {
            session_id: sorted(utterance_ids)
            for session_id, utterance_ids in sorted(sessions.items())
        }
        for speaker_id, sessions in sorted(pools.items())
    }
    return normalized, utterance_meta


def _qualifying_sessions(
    pools: dict[str, dict[str, list[str]]],
    minimum_utterances: int,
) -> dict[str, list[str]]:
    return {
        speaker_id: [
            session_id
            for session_id, utterance_ids in sessions.items()
            if len(utterance_ids) >= minimum_utterances
        ]
        for speaker_id, sessions in pools.items()
    }


def _write_seed_trials(
    *,
    pools: dict[str, dict[str, list[str]]],
    eligible: Sequence[str],
    qualifying: dict[str, list[str]],
    output_path: Path,
    split_name: str,
    n_values: Sequence[int],
    enrollment_n: int,
    seed: int,
) -> dict[str, object]:
    max_n = max(n_values)
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    selected: dict[str, tuple[str, str, list[str], list[str]]] = {}
    for speaker_id in eligible:
        target_session = rng.choice(qualifying[speaker_id])
        enrollment_candidates = [
            session_id
            for session_id in qualifying[speaker_id]
            if session_id != target_session
            and _call_id(session_id) != _call_id(target_session)
        ]
        if not enrollment_candidates:
            raise ValueError(f"speaker {speaker_id} has no call-disjoint enrollment session")
        enrollment_session = rng.choice(enrollment_candidates)
        enrollment_ids = rng.sample(pools[speaker_id][enrollment_session], enrollment_n)
        target_ids = rng.sample(pools[speaker_id][target_session], max_n)
        selected[speaker_id] = (
            enrollment_session,
            target_session,
            enrollment_ids,
            target_ids,
        )

    for speaker_id in eligible:
        enrollment_session, target_session, enrollment_ids, target_ids = selected[speaker_id]
        common = {
            "seed": seed,
            "split": split_name,
            "enrollment_n": enrollment_n,
            "n_values": list(n_values),
            "enroll_speaker": speaker_id,
            "enroll_session_id": enrollment_session,
            "enroll_utt_ids": enrollment_ids,
        }
        records.append({
            **common,
            "trial_id": f"{split_name}-seed{seed}-{speaker_id}-tar",
            "label": 1,
            "target_speaker": speaker_id,
            "target_session_id": target_session,
            "target_utt_ids": target_ids,
        })

        impostor_candidates = [
            candidate
            for candidate in eligible
            if candidate != speaker_id
            and any(
                _call_id(session_id) != _call_id(enrollment_session)
                for session_id in qualifying[candidate]
            )
        ]
        if not impostor_candidates:
            raise ValueError(f"speaker {speaker_id} has no call-disjoint impostor")
        target_speaker = rng.choice(impostor_candidates)
        impostor_sessions = [
            session_id
            for session_id in qualifying[target_speaker]
            if _call_id(session_id) != _call_id(enrollment_session)
        ]
        impostor_session = rng.choice(impostor_sessions)
        impostor_ids = rng.sample(pools[target_speaker][impostor_session], max_n)
        records.append({
            **common,
            "trial_id": f"{split_name}-seed{seed}-{speaker_id}-non",
            "label": 0,
            "target_speaker": target_speaker,
            "target_session_id": impostor_session,
            "target_utt_ids": impostor_ids,
        })

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "seed": seed,
        "trial_file": str(output_path.resolve()),
        "target_trials": len(eligible),
        "nontarget_trials": len(eligible),
        "trials": len(records),
    }


def validate_session_trials(
    path: str | Path,
    manifest_path: str | Path,
    n_values: Sequence[int],
    enrollment_n: int,
    *,
    _utterance_meta: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    ordered_n = sorted(set(int(value) for value in n_values))
    if not ordered_n or ordered_n[0] <= 0:
        raise ValueError("n_values must contain positive integers")
    max_n = ordered_n[-1]
    utterance_meta = _utterance_meta or {
        row["utt_id"]: (row["speaker_id"], fisher_session_id(row), row["call_id"])
        for row in iter_manifest(manifest_path)
    }
    counts = {"target": 0, "nontarget": 0}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            trial = json.loads(line)
            enroll = list(map(str, trial["enroll_utt_ids"]))
            target = list(map(str, trial["target_utt_ids"]))
            if len(enroll) != enrollment_n or len(set(enroll)) != enrollment_n:
                raise ValueError(f"trial line {line_number} has invalid fixed enrollment")
            if len(target) != max_n or len(set(target)) != max_n:
                raise ValueError(f"trial line {line_number} has invalid max-N target")
            missing = [utt_id for utt_id in enroll + target if utt_id not in utterance_meta]
            if missing:
                raise KeyError(f"trial line {line_number} references unknown utterance {missing[0]}")

            enroll_meta = {utterance_meta[utt_id] for utt_id in enroll}
            target_meta = {utterance_meta[utt_id] for utt_id in target}
            if {(speaker, session) for speaker, session, _ in enroll_meta} != {
                (trial["enroll_speaker"], trial["enroll_session_id"])
            }:
                raise ValueError(f"trial line {line_number} mixes enrollment sessions")
            if {(speaker, session) for speaker, session, _ in target_meta} != {
                (trial["target_speaker"], trial["target_session_id"])
            }:
                raise ValueError(f"trial line {line_number} mixes target sessions")
            enroll_calls = {call for _, _, call in enroll_meta}
            target_calls = {call for _, _, call in target_meta}
            if enroll_calls & target_calls:
                raise ValueError(f"trial line {line_number} has call leakage")
            label = int(trial["label"])
            if (label == 1) != (trial["enroll_speaker"] == trial["target_speaker"]):
                raise ValueError(f"trial line {line_number} has inconsistent label")
            for smaller, larger in zip(ordered_n, ordered_n[1:]):
                if target[:smaller] != target[:larger][:smaller]:
                    raise ValueError(f"trial line {line_number} is not nested")
            counts["target" if label == 1 else "nontarget"] += 1
    return counts


def build_session_trial_sets(
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    split_name: str,
    n_values: Sequence[int],
    enrollment_n: int,
    seeds: Sequence[int],
    allowed_utterance_ids: set[str] | None = None,
) -> dict[str, object]:
    ordered_n = sorted(set(int(value) for value in n_values))
    ordered_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if not ordered_n or ordered_n[0] <= 0:
        raise ValueError("n_values must contain positive integers")
    if enrollment_n <= 0:
        raise ValueError("enrollment_n must be positive")
    if not ordered_seeds:
        raise ValueError("at least one seed is required")

    pools, utterance_meta = _load_session_pools(
        manifest_path,
        split_path,
        split_name,
        allowed_utterance_ids,
    )
    minimum = max(max(ordered_n), enrollment_n)
    qualifying = _qualifying_sessions(pools, minimum)
    eligible = sorted(
        speaker_id
        for speaker_id, session_ids in qualifying.items()
        if len({_call_id(session_id) for session_id in session_ids}) >= 2
    )
    if len(eligible) < 2:
        raise ValueError(
            f"{split_name} has fewer than two speakers with two call-disjoint "
            f"sessions containing at least {minimum} utterances"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed_audits = []
    combined_path = destination / "all_seeds.jsonl"
    with combined_path.open("w", encoding="utf-8") as combined:
        for seed in ordered_seeds:
            output_path = destination / f"sampling_seed_{seed}.jsonl"
            seed_audits.append(_write_seed_trials(
                pools=pools,
                eligible=eligible,
                qualifying=qualifying,
                output_path=output_path,
                split_name=split_name,
                n_values=ordered_n,
                enrollment_n=enrollment_n,
                seed=seed,
            ))
            validate_session_trials(
                output_path,
                manifest_path,
                ordered_n,
                enrollment_n,
                _utterance_meta=utterance_meta,
            )
            combined.write(output_path.read_text(encoding="utf-8"))

    audit = {
        "protocol": "fixed-original-enrollment/increasing-anonymized-target",
        "session_definition": "fisher_call_id_plus_channel",
        "manifest": str(Path(manifest_path).resolve()),
        "splits": str(Path(split_path).resolve()),
        "split": split_name,
        "n_values": ordered_n,
        "nested_target_sampling": True,
        "enrollment_n": enrollment_n,
        "enrollment_fixed_across_n": True,
        "seeds": ordered_seeds,
        "eligible_speakers": len(eligible),
        "allowed_utterance_filter": allowed_utterance_ids is not None,
        "allowed_utterances": (
            None if allowed_utterance_ids is None else len(allowed_utterance_ids)
        ),
        "combined_trial_file": str(combined_path.resolve()),
        "per_seed": seed_audits,
    }
    (destination / "session_trials.audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def iter_target_utterance_ids(paths: Iterable[str | Path]) -> Iterable[str]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                trial = json.loads(line)
                yield from map(str, trial["target_utt_ids"])
