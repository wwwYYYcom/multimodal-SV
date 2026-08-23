from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


TRANSCRIPT_RE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s+(?P<end>\d+(?:\.\d+)?)\s+"
    r"(?P<channel>[AB]):\s*(?P<text>.*)\s*$"
)


@dataclass(frozen=True)
class FisherSegment:
    utt_id: str
    speaker_id: str
    call_id: str
    channel: int
    audio_path: str
    start: float
    end: float
    duration: float
    transcript: str


MANIFEST_FIELDS = [field.name for field in FisherSegment.__dataclass_fields__.values()]


def normalize_call_id(value: str) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        raise ValueError(f"无效 call id: {value!r}")
    return digits[-5:].zfill(5)


def read_call_speakers(calldata_path: str | Path) -> dict[tuple[str, str], str]:
    """从 LDC calldata 表读取 (call, channel) -> speaker PIN。"""
    mapping: dict[tuple[str, str], str] = {}
    with Path(calldata_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            call_id = normalize_call_id(row["CALL_ID"])
            mapping[(call_id, "A")] = str(row["APIN"]).strip()
            mapping[(call_id, "B")] = str(row["BPIN"]).strip()
    if not mapping:
        raise ValueError(f"calldata 为空: {calldata_path}")
    return mapping


def index_by_stem(root: str | Path, suffix: str) -> dict[str, Path]:
    paths = sorted(Path(root).rglob(f"*{suffix}"))
    index = {path.stem: path.resolve() for path in paths}
    if len(index) != len(paths):
        raise ValueError(f"{root} 中发现重复 stem，无法唯一匹配 {suffix}")
    return index


def parse_transcript(
    transcript_path: str | Path,
    audio_path: str | Path,
    call_speakers: dict[tuple[str, str], str],
    min_duration: float = 0.0,
) -> Iterator[FisherSegment]:
    transcript_path = Path(transcript_path)
    call_id = normalize_call_id(transcript_path.stem)
    per_channel_index: Counter[str] = Counter()

    with transcript_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = TRANSCRIPT_RE.match(line)
            if not match:
                continue
            start = float(match.group("start"))
            end = float(match.group("end"))
            channel_letter = match.group("channel")
            if end <= start:
                raise ValueError(f"结束时间不大于开始时间: {transcript_path}:{line_number}")
            duration = end - start
            if duration < min_duration:
                continue
            speaker_id = call_speakers.get((call_id, channel_letter))
            if not speaker_id:
                raise KeyError(f"calldata 缺少 call={call_id}, channel={channel_letter}")
            per_channel_index[channel_letter] += 1
            utt_id = f"fe_03_{call_id}_{channel_letter}_{per_channel_index[channel_letter]:04d}"
            yield FisherSegment(
                utt_id=utt_id,
                speaker_id=speaker_id,
                call_id=f"fe_03_{call_id}",
                channel=0 if channel_letter == "A" else 1,
                audio_path=str(Path(audio_path).resolve()),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                transcript=match.group("text").strip(),
            )


def build_manifest(
    audio_root: str | Path,
    transcript_root: str | Path,
    calldata_path: str | Path,
    output_csv: str | Path,
    min_duration: float = 1.0,
    max_calls: int | None = None,
) -> dict[str, object]:
    audio_index = index_by_stem(audio_root, ".sph")
    transcript_index = index_by_stem(transcript_root, ".txt")
    common_stems = sorted(audio_index.keys() & transcript_index.keys())
    if max_calls is not None:
        common_stems = common_stems[:max_calls]
    if not common_stems:
        raise FileNotFoundError("没有找到 stem 匹配的 .sph 与 transcript .txt")

    call_speakers = read_call_speakers(calldata_path)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    speaker_calls: dict[str, set[str]] = defaultdict(set)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for stem in common_stems:
            for segment in parse_transcript(
                transcript_index[stem], audio_index[stem], call_speakers, min_duration
            ):
                writer.writerow(asdict(segment))
                counts["utterances"] += 1
                counts[f"channel_{segment.channel}"] += 1
                speaker_calls[segment.speaker_id].add(segment.call_id)

    audit = {
        "manifest": str(output_path.resolve()),
        "audio_files": len(audio_index),
        "transcripts": len(transcript_index),
        "matched_calls": len(common_stems),
        "missing_audio_for_transcript": len(transcript_index.keys() - audio_index.keys()),
        "missing_transcript_for_audio": len(audio_index.keys() - transcript_index.keys()),
        "utterances": counts["utterances"],
        "speakers": len(speaker_calls),
        "speakers_with_at_least_2_calls": sum(len(calls) >= 2 for calls in speaker_calls.values()),
        "min_duration": min_duration,
    }
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_manifest(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)

