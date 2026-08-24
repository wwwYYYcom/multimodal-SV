import numpy as np

from mmsv.audio import crop_or_repeat


def test_crop_or_repeat_fills_short_audio_without_zeros():
    waveform = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    output = crop_or_repeat(waveform, 8)
    assert output.dtype == np.float32
    assert output.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]


def test_crop_or_repeat_honors_start_for_long_audio():
    waveform = np.arange(10, dtype=np.float32)
    assert crop_or_repeat(waveform, 4, start=3).tolist() == [3.0, 4.0, 5.0, 6.0]
