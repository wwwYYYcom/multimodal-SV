from __future__ import annotations

import io
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def find_sph2pipe(explicit: str | None = None) -> str | None:
    candidates = [explicit, os.environ.get("SPH2PIPE"), shutil.which("sph2pipe")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return None


def _read_with_soundfile(source: str | Path | io.BytesIO) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("读取音频需要 soundfile") from exc
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    return waveform, int(sample_rate)


@lru_cache(maxsize=4)
def _decode_sphere_cached(path: str) -> tuple[np.ndarray, int]:
    """缓存最近 call；同一 call 的 A/B 与相邻 turns 只做一次 Shorten 解码。"""
    from desphere import transcode_bytes

    wav_bytes = transcode_bytes(Path(path).read_bytes())
    return _read_with_soundfile(io.BytesIO(wav_bytes))


def read_audio(
    path: str | Path,
    channel: int = 0,
    sph2pipe: str | None = None,
    start: float | None = None,
    end: float | None = None,
) -> tuple[np.ndarray, int]:
    """返回单通道 float32 waveform；SPHERE 的 channel 使用 0/1 索引。"""
    audio_path = Path(path)
    if audio_path.suffix.lower() != ".sph":
        waveform, sample_rate = _read_with_soundfile(audio_path)
        if channel >= waveform.shape[1]:
            raise ValueError(f"请求 channel={channel}，音频只有 {waveform.shape[1]} 个通道")
        return waveform[:, channel], sample_rate

    executable = find_sph2pipe(sph2pipe)
    if executable:
        command = [executable, "-f", "wav", "-p", "-c", str(channel + 1)]
        if start is not None or end is not None:
            command.extend(["-t", f"{'' if start is None else start}:{'' if end is None else end}"])
        command.append(str(audio_path))
        process = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"sph2pipe 失败 ({process.returncode}): {stderr}")
        waveform, sample_rate = _read_with_soundfile(io.BytesIO(process.stdout))
        return waveform[:, 0], sample_rate

    # Windows 无现成 sph2pipe 时使用 MIT 许可的 clean-room Shorten 解码器。
    try:
        from desphere import transcode_bytes
    except ImportError as exc:
        raise FileNotFoundError(
            "找不到 sph2pipe，也未安装 desphere[fast]。执行 pip install 'desphere[fast]'。"
        ) from exc
    waveform, sample_rate = _decode_sphere_cached(str(audio_path.resolve()))
    if channel >= waveform.shape[1]:
        raise ValueError(f"请求 channel={channel}，SPHERE 只有 {waveform.shape[1]} 个通道")
    begin = 0 if start is None else max(0, round(start * sample_rate))
    finish = len(waveform) if end is None else min(len(waveform), round(end * sample_rate))
    return waveform[begin:finish, channel], sample_rate


def read_segment(
    row: dict[str, str], target_sample_rate: int = 16000, sph2pipe: str | None = None
) -> np.ndarray:
    start_seconds = float(row.get("start", 0.0))
    end_raw = row.get("end")
    end_seconds = None if end_raw in (None, "") else float(end_raw)
    waveform, sample_rate = read_audio(
        row["audio_path"], int(row.get("channel", 0)), sph2pipe, start_seconds, end_seconds
    )
    segment = waveform
    if segment.size == 0:
        raise ValueError(f"空音频段: {row.get('utt_id', '<unknown>')}")
    if sample_rate != target_sample_rate:
        divisor = int(np.gcd(sample_rate, target_sample_rate))
        segment = resample_poly(segment, target_sample_rate // divisor, sample_rate // divisor)
    return np.asarray(segment, dtype=np.float32)


def crop_or_pad(waveform: np.ndarray, length: int, start: int | None = None) -> np.ndarray:
    if waveform.size >= length:
        if start is None:
            start = 0
        start = max(0, min(int(start), waveform.size - length))
        return waveform[start : start + length]
    output = np.zeros(length, dtype=np.float32)
    output[: waveform.size] = waveform
    return output


def crop_or_repeat(waveform: np.ndarray, length: int, start: int | None = None) -> np.ndarray:
    """裁剪长语音；短语音循环填充，避免把大段零静音送入 speaker encoder。"""
    if waveform.size == 0:
        raise ValueError("不能循环填充空音频")
    if waveform.size >= length:
        if start is None:
            start = 0
        start = max(0, min(int(start), waveform.size - length))
        return np.asarray(waveform[start : start + length], dtype=np.float32)
    repeats = (length + waveform.size - 1) // waveform.size
    return np.tile(waveform, repeats)[:length].astype(np.float32, copy=False)
