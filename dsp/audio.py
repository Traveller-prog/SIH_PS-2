"""Analog audio demodulation (AM, narrow/wide FM) of complex baseband IQ for listening."""

from fractions import Fraction

import numpy as np
from scipy import signal

AUDIO_MODES = ("AM", "NFM", "WFM")
AUDIO_RATE = 48_000

# FM de-emphasis time constants (seconds); NFM voice radio uses none
_DEEMPHASIS = {"WFM": 75e-6}


def _to_audio_rate(x, fs, audio_rate):
    ratio = Fraction(audio_rate / fs).limit_denominator(2000)
    if ratio == 1:
        return x
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def demodulate_audio(iq, sample_rate, mode="AM", audio_rate=AUDIO_RATE, peak=0.9):
    """Demodulate a channel centred at 0 Hz to mono audio.

    Args:
        iq: Complex baseband samples containing just the channel of interest (band-limit
            with dsp.spectral.extract_band first when the recording holds more).
        sample_rate: Rate of iq in Hz.
        mode: 'AM' (envelope), 'NFM' or 'WFM' (quadrature discriminator; WFM adds 75 us
            de-emphasis).
        audio_rate: Output rate in Hz.
        peak: Output peak level (0..1); silence stays silent.

    Returns:
        (audio float32 in [-peak, peak], audio_rate)
    """
    mode = mode.upper()
    if mode not in AUDIO_MODES:
        raise ValueError(f"mode must be one of {AUDIO_MODES}")
    x = np.asarray(iq, dtype=np.complex64)
    if x.size < 64:
        raise ValueError("Need at least 64 samples to demodulate audio")
    if sample_rate < audio_rate / 4:
        raise ValueError("Sample rate is too low for audio")

    if mode == "AM":
        a = np.abs(x).astype(np.float64)
    else:
        # instantaneous frequency (Hz): phase step of consecutive samples
        a = np.angle(x[1:] * np.conj(x[:-1])).astype(np.float64) * sample_rate / (2 * np.pi)
        a = np.concatenate([a[:1], a])
    a = a - np.mean(a)  # carrier level (AM) or carrier offset (FM)

    audio = _to_audio_rate(a, sample_rate, audio_rate)
    if mode in _DEEMPHASIS:
        alpha = 1.0 - np.exp(-1.0 / (audio_rate * _DEEMPHASIS[mode]))
        audio = signal.lfilter([alpha], [1.0, alpha - 1.0], audio)
    audio = audio - np.mean(audio)
    top = np.max(np.abs(audio))
    if top > 0:
        audio = audio * (peak / top)
    return audio.astype(np.float32), audio_rate


def to_pcm16(audio):
    """float audio in [-1, 1] -> little-endian int16 bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
