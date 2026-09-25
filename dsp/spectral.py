"""Spectral analysis helpers for complex IQ arrays."""

import numpy as np
from scipy import signal

_EPS = 1e-20


def compute_psd(iq, sample_rate, nperseg=1024, window="hann", noverlap=None, db=True):
    """Welch PSD of a complex IQ array, with a centered (negative..positive) frequency axis.

    Args:
        iq: Complex samples.
        sample_rate: Sampling rate in Hz.
        nperseg: Segment length (clamped to the signal length).
        window: Window name or array accepted by scipy.signal.welch.
        noverlap: Overlap in samples (default nperseg // 2).
        db: Return 10*log10(PSD) if True, else linear power/Hz.

    Returns:
        (freqs_hz, psd): both float64 arrays, ascending frequency.
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        raise ValueError("iq is empty")
    nperseg = min(int(nperseg), iq.size)
    freqs, psd = signal.welch(
        iq,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        detrend=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    if db:
        psd = 10.0 * np.log10(np.maximum(psd, _EPS))
    return freqs, psd


def compute_stft(iq, sample_rate, nperseg=1024, noverlap=None, window="hann", db=True):
    """STFT magnitude matrix for a waterfall plot.

    Returns:
        (freqs_hz, times_s, matrix): matrix has shape (n_freqs, n_times) with
        frequency ascending (centered). Values are power in dB if db else magnitude.
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        raise ValueError("iq is empty")
    nperseg = min(int(nperseg), iq.size)
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, times, zxx = signal.stft(
        iq,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    freqs = np.fft.fftshift(freqs)
    mag = np.abs(np.fft.fftshift(zxx, axes=0))
    if db:
        mag = 20.0 * np.log10(np.maximum(mag, _EPS))
    return freqs, times, mag


def compute_constellation(iq, max_points=5000):
    """IQ scatter points normalized so that all coordinates lie in [-1, 1].

    The DC offset is left intact; scaling divides by the peak |I| or |Q| so the
    aspect ratio is preserved. Long inputs are evenly decimated to max_points.

    Returns:
        (i, q): float32 arrays.
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        raise ValueError("iq is empty")
    if max_points and iq.size > max_points:
        idx = np.linspace(0, iq.size - 1, int(max_points)).astype(np.int64)
        iq = iq[idx]
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)
    peak = max(float(np.max(np.abs(i))), float(np.max(np.abs(q))))
    if peak > 0:
        i = i / peak
        q = q / peak
    return i, q


def _cyclic_spectrum(feature, seg_len):
    """Averaged, windowed periodogram of a real feature (mean removed)."""
    n = len(feature)
    seg_len = min(seg_len, n)
    win = np.hanning(seg_len)
    acc = np.zeros(seg_len // 2 + 1)
    count = 0
    for start in range(0, n - seg_len + 1, seg_len // 2):
        seg = feature[start : start + seg_len]
        seg = (seg - seg.mean()) * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
        count += 1
    return acc / max(count, 1), seg_len


def estimate_symbol_rate(iq, sample_rate, max_samples=262144, seg_len=16384,
                         min_sps=2.0, max_sps=64.0, min_score_db=13.0):
    """Estimate the symbol (baud) rate from cyclostationary spectral lines.

    Two cycle features are examined: the instantaneous power |x|^2 (linear
    modulations with excess bandwidth: PSK/QAM) and the magnitude of the change in
    instantaneous frequency (constant-envelope FSK). The strongest line, measured
    against the median spectrum level, gives the baud rate; the peak is refined with
    parabolic interpolation.

    Returns:
        (baud_hz, score_db): score is the peak-to-median ratio in dB.

    Raises:
        ValueError if the signal is too short or no convincing spectral line exists.
    """
    x = np.asarray(iq, dtype=np.complex128)[:max_samples]
    if x.size < 512:
        raise ValueError("Need at least 512 samples to estimate the symbol rate")

    inst_freq = np.angle(x[1:] * np.conj(x[:-1]))
    features = (np.abs(x) ** 2, np.abs(np.diff(inst_freq)))

    best = None
    for feat_idx, feat in enumerate(features):
        spec, n = _cyclic_spectrum(feat, seg_len)
        lo = max(int(np.ceil(n / max_sps)), 3)
        hi = min(int(np.floor(n / min_sps)), len(spec) - 2)
        if hi <= lo:
            continue
        band = spec[lo : hi + 1]
        floor = max(np.median(band), 1e-30)
        k = int(np.argmax(band)) + lo
        score = 10 * np.log10(spec[k] / floor)
        # FSK frequency-change feature: a comparable line at f/m means this one is a harmonic.
        for m in (4, 3, 2) if feat_idx == 1 else ():
            km = int(round(k / m))
            if km - 2 < lo:
                continue
            j = km - 2 + int(np.argmax(spec[km - 2 : km + 3]))
            sub_score = 10 * np.log10(spec[j] / floor)
            if sub_score >= max(min_score_db, score - 12.0):
                k, score = j, sub_score
                break
        # Parabolic refinement on the log spectrum.
        a, b, c = np.log(spec[k - 1 : k + 2] + 1e-30)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        baud = (k + delta) * sample_rate / n
        if best is None or score > best[1]:
            best = (baud, score)

    if best is None or best[1] < min_score_db:
        raise ValueError("No symbol-rate spectral line found; enter the symbol rate manually")
    return float(best[0]), float(best[1])


def estimate_snr(iq, sample_rate, nperseg=1024, noise_fraction=0.1, band_factor=2.0):
    """In-band SNR (dB) from the PSD: noise floor from the quietest bins, signal from the bins above it.

    The floor is the mean of the lowest `noise_fraction` of bins; bins above band_factor
    times the floor form the occupied band, and SNR = (band power - floor*band) / (floor*band).
    Needs some empty spectrum around the signal; raises ValueError when the signal fills
    the whole band or no signal stands out.
    """
    _, psd = compute_psd(iq, sample_rate, nperseg=nperseg, db=False)
    order = np.sort(psd)
    floor = float(np.mean(order[: max(int(len(order) * noise_fraction), 1)]))
    band = psd > band_factor * floor
    if not band.any():
        raise ValueError("No signal stands out above the noise floor")
    if band.mean() > 0.9:
        raise ValueError("Signal fills the whole band; no noise reference")
    excess = float(np.sum(psd[band] - floor))
    noise = floor * band.sum()
    return 10.0 * np.log10(max(excess, _EPS) / noise)


# Kurtosis E|s|^4 / E|s|^2^2 of the unit-power constellations, for the M2M4 SNR estimator.
CONSTELLATION_KURTOSIS = {"BPSK": 1.0, "QPSK": 1.0, "8-PSK": 1.0, "16-QAM": 1.32, "64-QAM": 1.381}


def estimate_cfo(iq, sample_rate, order=4, max_samples=262144, min_score_db=15.0):
    """Carrier frequency offset by the M-th power method (FFT peak search on x**M).

    Raising a linearly modulated signal to M (2 BPSK, 4 QPSK/QAM, 8 8-PSK) strips the data
    and leaves a tone at M*offset. The peak is found on a zero-padded FFT and refined by
    parabolic interpolation. Unambiguous range: |offset| < sample_rate / (2*order).

    Returns:
        (cfo_hz, score_db): score is the tone's peak-to-median level; raises ValueError
        when no tone stands out (min_score_db), e.g. FSK, noise or very short input.
    """
    x = np.asarray(iq, dtype=np.complex128)[:max_samples]
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    if x.size < 64:
        raise ValueError("Need at least 64 samples to estimate the carrier offset")
    xm = x ** order
    nfft = 1 << int(np.ceil(np.log2(x.size * 4)))
    spec = np.abs(np.fft.fft(xm * np.hanning(x.size), nfft)) ** 2
    k = int(np.argmax(spec))
    score_db = 10 * np.log10(spec[k] / max(np.median(spec), _EPS))
    if score_db < min_score_db:
        raise ValueError("No carrier tone found; the modulation order may be wrong")
    a, b, c = (np.log(spec[(k + d) % nfft] + _EPS) for d in (-1, 0, 1))
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    bin_pos = k + float(np.clip(delta, -0.5, 0.5))
    if bin_pos > nfft / 2:
        bin_pos -= nfft
    return float(bin_pos * sample_rate / nfft / order), float(score_db)


def correct_cfo(iq, sample_rate, order=4, cfo_hz=None, remove_phase=False, **kwargs):
    """Remove the carrier frequency offset so the constellation stops spinning.

    Estimates the offset with estimate_cfo unless cfo_hz is given, then multiplies by
    exp(-j*2*pi*cfo*t). With remove_phase=True the static phase is also derotated using the
    M-th power angle (leaves the usual 2*pi/M ambiguity; QAM lands on a fixed extra rotation).

    Returns:
        (corrected_iq, cfo_hz)
    """
    x = np.asarray(iq)
    if cfo_hz is None:
        cfo_hz, _ = estimate_cfo(x, sample_rate, order, **kwargs)
    y = x * np.exp(-2j * np.pi * cfo_hz / sample_rate * np.arange(x.size))
    if remove_phase:
        y = y * np.exp(-1j * np.angle(np.sum(y ** order)) / order)
    return y.astype(x.dtype if np.iscomplexobj(x) else np.complex64, copy=False), float(cfo_hz)


def estimate_snr_m2m4(iq, modulation=None, kurtosis=None):
    """SNR in dB by the M2M4 moment method (blind, invariant to carrier phase/frequency).

    With M2 = E|x|^2 and M4 = E|x|^4 for signal (kurtosis ka) plus complex Gaussian noise
    (kurtosis 2):  S = sqrt((2*M2^2 - M4) / (2 - ka)),  N = M2 - S.

    Feed symbol-spaced (ideally matched-filtered) samples: pulse-shaped, oversampled data
    does not have the constellation's kurtosis and biases the result. Pass modulation
    (a key of CONSTELLATION_KURTOSIS) or kurtosis directly; the default is constant
    modulus (1.0).

    Returns:
        SNR in dB; raises ValueError when the moments are inconsistent (SNR too low for
        the sample count, or the samples are not a signal plus Gaussian noise).
    """
    x = np.asarray(iq, dtype=np.complex128)
    if x.size < 100:
        raise ValueError("Need at least 100 samples for M2M4 SNR estimation")
    ka = kurtosis if kurtosis is not None else CONSTELLATION_KURTOSIS.get(modulation, 1.0)
    if ka >= 2:
        raise ValueError("kurtosis must be below 2 (Gaussian) for M2M4")
    p = np.abs(x) ** 2
    m2, m4 = float(p.mean()), float(np.mean(p ** 2))
    val = (2 * m2 ** 2 - m4) / (2 - ka)
    if val <= 0:
        raise ValueError("M2M4 estimate invalid (noise dominates or samples are not symbol-spaced)")
    s = np.sqrt(val)
    n = m2 - s
    if n <= 0:
        return float("inf")
    return float(10 * np.log10(s / n))


# --- Streaming variants: read only the frames they need from a lazy source (dsp.file_reader.IQSource).
# A source needs len(source) and read_frames(starts, length).

def _frame_starts(n_samples, length, max_frames, min_hop=None):
    """Evenly spaced frame start indices covering the file, at most max_frames of them."""
    last = n_samples - length
    hop = max(min_hop or length // 2, int(np.ceil(last / max(max_frames - 1, 1))) if last > 0 else 1, 1)
    return np.arange(0, last + 1, hop, dtype=np.int64)


def compute_psd_from_source(source, sample_rate, nperseg=1024, max_frames=2048, window="hann",
                            db=True, batch=256):
    """Welch-style PSD averaged over frames spread evenly across the whole file.

    Only max_frames * nperseg samples are ever read, so large memory-mapped recordings stay
    cheap. Same scaling as compute_psd. Returns (freqs_hz, psd).
    """
    n = len(source)
    if n == 0:
        raise ValueError("source is empty")
    nperseg = min(int(nperseg), n)
    starts = _frame_starts(n, nperseg, max_frames)
    w = signal.get_window(window, nperseg)
    acc = np.zeros(nperseg)
    for i in range(0, len(starts), batch):
        frames = source.read_frames(starts[i : i + batch], nperseg)
        acc += np.sum(np.abs(np.fft.fft(frames * w, axis=1)) ** 2, axis=0)
    psd = np.fft.fftshift(acc / (len(starts) * sample_rate * np.sum(w ** 2)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, 1.0 / sample_rate))
    if db:
        psd = 10.0 * np.log10(np.maximum(psd, _EPS))
    return freqs, psd


def compute_stft_from_source(source, sample_rate, nperseg=1024, max_frames=400, window="hann", db=True):
    """Waterfall matrix from up to max_frames frames spread evenly across the file.

    Same output as compute_stft; the time axis spans the whole file with a hop of
    max(nperseg/2, n/max_frames) samples, and only those frames are read from disk.
    """
    n = len(source)
    if n == 0:
        raise ValueError("source is empty")
    nperseg = min(int(nperseg), n)
    starts = _frame_starts(n, nperseg, max_frames)
    w = signal.get_window(window, nperseg)
    mags = np.empty((nperseg, len(starts)))
    for i in range(0, len(starts), 256):
        frames = source.read_frames(starts[i : i + 256], nperseg)
        mags[:, i : i + len(frames)] = (np.abs(np.fft.fft(frames * w, axis=1)) / np.sum(w)).T
    mags = np.fft.fftshift(mags, axes=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, 1.0 / sample_rate))
    times = (starts + nperseg / 2) / sample_rate
    if db:
        mags = 20.0 * np.log10(np.maximum(mags, _EPS))
    return freqs, times, mags
