from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def count_syllables(words: Iterable[str]) -> int:
    """使用 CMUdict；词典外单词使用元音组启发式，并把计数下限设为 1。"""
    try:
        import pronouncing
    except ImportError as exc:
        raise RuntimeError("韵律分支需要 pronouncing") from exc

    total = 0
    for raw_word in words:
        word = "".join(character for character in raw_word.lower() if character.isalpha())
        if not word:
            continue
        phones = pronouncing.phones_for_word(word)
        if phones:
            total += pronouncing.syllable_count(phones[0])
            continue
        groups = 0
        previous_vowel = False
        for character in word:
            is_vowel = character in "aeiouy"
            groups += int(is_vowel and not previous_vowel)
            previous_vowel = is_vowel
        total += max(1, groups)
    return total


def extract_prosody(
    waveform: np.ndarray,
    sample_rate: int,
    word_timestamps: Sequence[dict[str, float | str]],
) -> np.ndarray:
    """返回论文定义的 [F0_mean, speaking_rate, voiced_ratio]。"""
    try:
        import parselmouth
    except ImportError as exc:
        raise RuntimeError("韵律分支需要 praat-parselmouth") from exc

    sound = parselmouth.Sound(np.asarray(waveform, dtype=np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch()
    frequencies = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    voiced = frequencies > 0
    f0_mean = float(frequencies[voiced].mean()) if voiced.any() else 0.0
    voiced_ratio = float(voiced.mean()) if frequencies.size else 0.0

    usable_words = [
        item for item in word_timestamps
        if "word" in item and "start" in item and "end" in item
    ]
    if usable_words:
        duration = max(
            1.0e-6,
            float(usable_words[-1]["end"]) - float(usable_words[0]["start"]),
        )
        speaking_rate = count_syllables(str(item["word"]) for item in usable_words) / duration
    else:
        speaking_rate = 0.0
    return np.asarray([f0_mean, speaking_rate, voiced_ratio], dtype=np.float32)


def summarize_prosody(features: torch.Tensor) -> torch.Tensor:
    """N x 3 -> 6，使用 population std（correction=0）避免 N=1 NaN。"""
    if features.ndim < 2 or features.shape[-1] != 3:
        raise ValueError("prosody features 最后一维必须为 3")
    return torch.cat([features.mean(dim=-2), features.std(dim=-2, correction=0)], dim=-1)


class AudioTextFusion(nn.Module):
    """论文 III-A2：两模态各投影到 256，LayerNorm，拼接后输出 192。"""

    def __init__(self, audio_dim: int = 192, text_dim: int = 512, hidden_dim: int = 256, output_dim: int = 192):
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.text = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(torch.cat([self.audio(audio), self.text(text)], dim=-1)), dim=-1)


class AudioProsodyFusion(nn.Module):
    def __init__(self, audio_dim: int = 192, prosody_dim: int = 3, hidden_dim: int = 64, output_dim: int = 192):
        super().__init__()
        self.prosody = nn.Sequential(nn.Linear(prosody_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(audio_dim + hidden_dim, output_dim)

    def forward(self, audio: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(torch.cat([audio, self.prosody(prosody)], dim=-1)), dim=-1)


def weighted_fusion(audio: torch.Tensor, text: torch.Tensor, audio_weight: float) -> torch.Tensor:
    if audio.shape != text.shape:
        raise ValueError("加权融合前 audio/text 必须同形")
    if not 0.0 <= audio_weight <= 1.0:
        raise ValueError("audio_weight 必须在 [0,1]")
    return F.normalize(audio_weight * audio + (1.0 - audio_weight) * text, dim=-1)
