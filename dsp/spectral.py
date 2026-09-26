"""Spectral analysis helpers for complex IQ arrays."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy import ndimage, signal

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
    freqs = sfft.fftshift(freqs)
    psd = sfft.fftshift(psd)
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
    freqs = sfft.fftshift(freqs)
    mag = np.abs(sfft.fftshift(zxx, axes=0))
    if db:
        mag = 20.0 * np.log10(np.maximum(mag, _EPS))
    return freqs, times, mag


def compute_constellation(iq, max_points=2000):
    """IQ scatter points normalized so that all coordinates lie in [-1, 1].

    The DC offset is left intact; scaling divides by the peak |I| or |Q| so the
    aspect ratio is preserved. Long inputs are thinned with a stride slice (a view, no copy) so
    at most max_points points remain.

    Returns:
        (i, q): float32 arrays.
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        raise ValueError("iq is empty")
    if max_points and iq.size > max_points:
        iq = iq[:: -(-iq.size // int(max_points))]
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)
    peak = max(float(np.max(np.abs(i))), float(np.max(np.abs(q))))
    if peak > 0:
        i = i / peak
        q = q / peak
    return i, q


def _cyclic_spectrum(feature, seg_len):
    """Averaged, windowed periodogram of a real feature (mean removed per segment).

    The 50 %-overlapping segments are a zero-copy strided view of the feature; the mean removal
    and window are applied in place on the one temporary copy, and a single batched rFFT
    (multi-threaded) replaces the per-segment loop.
    """
    feature = np.asarray(feature, dtype=np.float64)
    seg_len = min(seg_len, len(feature))
    segs = sliding_window_view(feature, seg_len)[:: max(seg_len // 2, 1)]
    work = segs - segs.mean(axis=1, keepdims=True)
    work *= np.hanning(seg_len)
    spec = sfft.rfft(work, axis=1, overwrite_x=True, workers=-1)
    return (spec.real ** 2 + spec.imag ** 2).mean(axis=0), seg_len


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
    z = sfft.fft(xm * np.hanning(x.size), nfft, workers=-1)
    spec = z.real ** 2 + z.imag ** 2  # |z|^2 without the square root
    k = int(np.argmax(spec))
    score_db = 10 * np.log10(max(spec[k], _EPS) / max(np.median(spec), _EPS))
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
    y = shift_frequency(x, -cfo_hz / sample_rate)
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


def _frame_spectra(source, starts, nperseg, window, batch=256):
    """Yield (first_index, complex64 spectra) for frames of `source`, `batch` frames at a time.

    One pre-allocated (batch, nperseg) buffer is reused for every batch: the window is applied
    into it with out=, and the single-precision FFT runs in place (overwrite_x) on all cores.
    The batching only bounds memory; each batch is fully vectorized.
    """
    w = np.asarray(window, dtype=np.float32)
    buf = np.empty((min(batch, len(starts)), nperseg), dtype=np.complex64)
    for i in range(0, len(starts), batch):
        frames = source.read_frames(starts[i : i + batch], nperseg)
        view = buf[: len(frames)]
        np.multiply(frames, w, out=view)
        yield i, sfft.fft(view, axis=1, overwrite_x=True, workers=-1)


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
    for _, spec in _frame_spectra(source, starts, nperseg, w, batch):
        acc += (spec.real ** 2 + spec.imag ** 2).sum(axis=0, dtype=np.float64)
    psd = sfft.fftshift(acc / (len(starts) * sample_rate * np.sum(w ** 2)))
    freqs = sfft.fftshift(sfft.fftfreq(nperseg, 1.0 / sample_rate))
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
    for i, spec in _frame_spectra(source, starts, nperseg, w):
        mags[:, i : i + len(spec)] = np.hypot(spec.real, spec.imag).T / np.sum(w)
    mags = sfft.fftshift(mags, axes=0)
    freqs = sfft.fftshift(sfft.fftfreq(nperseg, 1.0 / sample_rate))
    times = (starts + nperseg / 2) / sample_rate
    if db:
        mags = 20.0 * np.log10(np.maximum(mags, _EPS))
    return freqs, times, mags


def shift_frequency(x, cycles_per_sample, block=4096):
    """Multiply x by exp(2j*pi*cycles_per_sample*n) (a frequency shift) without a full-length exp.

    The rotator is built as a `block`-long ramp times one phasor per block, so only
    block + len(x)/block complex exponentials are evaluated instead of len(x), and the
    multiplication runs on a reshaped view of x. Returns complex64.
    """
    x = np.asarray(x, dtype=np.complex64)
    n = x.size
    n_blocks = -(-n // block)
    padded = x if n == n_blocks * block else np.pad(x, (0, n_blocks * block - n))
    w = 2.0 * np.pi * cycles_per_sample
    ramp = np.exp(1j * w * np.arange(block)).astype(np.complex64)
    steps = np.exp(1j * w * block * np.arange(n_blocks)).astype(np.complex64)
    out = padded.reshape(n_blocks, block) * ramp
    out *= steps[:, None]
    return out.reshape(-1)[:n]


def extract_band(iq, sample_rate, f_lo, f_hi, oversample=6.0, min_out=1024, max_taps=4095):
    """Isolate the band [f_lo, f_hi] (Hz, relative to centre) as a complex baseband signal.

    Mixes the band centre to 0 Hz, low-pass filters to the band width and decimates so the
    output rate is about oversample x the bandwidth (less if that would leave fewer than
    min_out samples). A band covering (nearly) the whole spectrum is returned unchanged.

    Returns:
        (samples complex64, new_sample_rate, decimation)
    """
    x = np.asarray(iq, dtype=np.complex64)
    fs = float(sample_rate)
    lo, hi = max(float(f_lo), -fs / 2), min(float(f_hi), fs / 2)
    if hi <= lo:
        raise ValueError("Empty frequency range")
    bw, fc = hi - lo, (hi + lo) / 2
    if bw >= 0.9 * fs and abs(fc) < 0.05 * fs:
        return x, fs, 1
    if x.size < 64:
        raise ValueError("Too few samples in the selected time range")
    x = shift_frequency(x, -fc / fs)
    taps = int(np.clip(6 * fs / bw, 31, max_taps)) | 1
    h = signal.firwin(taps, min(bw / 2, 0.49 * fs), fs=fs)
    x = signal.oaconvolve(x, h, mode="same")
    decim = int(max(1, min(fs / (oversample * bw), x.size // min_out)))
    return x[::decim].astype(np.complex64, copy=False), fs / decim, decim  # [::decim] is a strided view


frame_starts = _frame_starts  # public alias for callers that need to know how many frames a streaming call reads


def compute_stft_row(iq, sample_rate, nperseg=1024, window="hann", db=True):
    """One waterfall row from a block of samples: the power averaged over all frames in the block.

    Same scaling as compute_stft (magnitude in dB). Returns (freqs_hz, row).
    """
    x = np.asarray(iq)
    nperseg = min(int(nperseg), x.size)
    if nperseg < 8:
        raise ValueError("Need at least 8 samples")
    n_frames = x.size // nperseg
    frames = x[: n_frames * nperseg].reshape(n_frames, nperseg)
    w = signal.get_window(window, nperseg)
    power = np.mean(np.abs(sfft.fft(frames * w, axis=1) / np.sum(w)) ** 2, axis=0)
    row = sfft.fftshift(np.sqrt(power))
    freqs = sfft.fftshift(sfft.fftfreq(nperseg, 1.0 / sample_rate))
    return freqs, (20.0 * np.log10(np.maximum(row, _EPS)) if db else row)


def detect_channels(freqs, psd_db, threshold_db=8.0, train_bins=None, percentile=20.0, min_bins=3,
                    merge_gap_hz=None, edge_db=3.0, max_channels=32):
    """Energy detector / blind channelizer: find every active signal in a PSD.

    A CFAR-style detector: the local noise floor at each bin is an order statistic (the
    `percentile`-th value) of a wide sliding window, which stays low even when a signal
    sits inside the window, so neighbouring signals do not mask each other. The floor is
    calibrated so noise-only bins sit at 0 dB; bins more than `threshold_db` above it are
    active. Active bins are merged across gaps (FSK tones, spectral nulls), short blips are
    dropped, and each channel's edges are grown outward to where it falls within `edge_db`
    of the floor.

    Args:
        freqs, psd_db: ascending frequency axis (Hz) and PSD in dB.
        threshold_db: detection margin over the noise floor (higher = fewer false alarms).
        train_bins: sliding-window length in bins (default a quarter of the spectrum).
        min_bins: narrowest channel kept (rejects DC spikes and noise blips).
        merge_gap_hz: gaps up to this width are bridged (default 2 % of the span).
        max_channels: keep only the strongest this many.

    Returns:
        List of dicts sorted by centre frequency: center_hz, bandwidth_hz, f_lo, f_hi,
        peak_hz, peak_db, snr_db (in-channel signal-to-noise ratio).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    power = 10.0 ** (np.asarray(psd_db, dtype=np.float64) / 10.0)
    n = power.size
    if n < 32:
        raise ValueError("Need at least 32 PSD bins to detect channels")
    df = float(freqs[1] - freqs[0])
    smooth = ndimage.uniform_filter1d(power, 3, mode="nearest")
    train = int(train_bins or max(n // 4, 31)) | 1
    floor = np.maximum(ndimage.percentile_filter(smooth, percentile, size=train, mode="nearest"), 1e-30)
    ratio = smooth / floor
    calib = float(np.median(ratio))  # noise-only bins should read 1
    ratio = ratio / calib
    floor_mean = floor * calib

    active = ratio > 10.0 ** (threshold_db / 10.0)
    gap = max(1, int(round((merge_gap_hz if merge_gap_hz is not None else 0.02 * n * df) / df)))

    # runs of active bins -> [start, stop); bridge gaps up to `gap` bins by grouping runs
    edges = np.flatnonzero(np.diff(active.astype(np.int8), prepend=0, append=0))
    starts, stops = edges[::2], edges[1::2]
    if starts.size == 0:
        return []
    first = np.flatnonzero(np.r_[True, (starts[1:] - stops[:-1]) > gap])
    last = np.r_[first[1:] - 1, starts.size - 1]
    delta = np.zeros(n + 1, dtype=np.int8)  # seeds as a boolean mask via a cumulative difference
    delta[starts[first]] = 1
    delta[stops[last]] = -1
    seed = np.cumsum(delta[:-1]) > 0

    # grow every seed outward through contiguous bins above the edge level: those are exactly the
    # connected components of (seed | ratio > grow) that contain a seed
    labels, _ = ndimage.label(seed | (ratio > 10.0 ** (edge_db / 10.0)))
    region = np.isin(labels, np.unique(labels[seed]))
    run_edges = np.flatnonzero(np.diff(region.astype(np.int8), prepend=0, append=0))
    a, b = run_edges[::2], run_edges[1::2]
    wide = (b - a) >= min_bins
    a, b = a[wide], b[wide]
    if a.size == 0:
        return []

    # per-channel statistics from cumulative sums / labelled reductions (no per-channel loop)
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v)])
    excess = cum(np.maximum(smooth - floor_mean, 0.0))
    noise = cum(floor_mean)
    snr = 10 * np.log10(np.maximum(excess[b] - excess[a], 1e-30) / (noise[b] - noise[a]))
    lab_runs, _ = ndimage.label(region)
    peak_bin = np.array(ndimage.maximum_position(power, lab_runs, np.arange(1, lab_runs.max() + 1)))[:, 0]
    peak_bin = peak_bin[wide]
    lo, hi = freqs[a] - df / 2, freqs[b - 1] + df / 2
    keep = np.argsort(-snr, kind="stable")[:max_channels]
    keep = keep[np.argsort((lo + hi)[keep], kind="stable")]
    return [
        {"center_hz": float((lo[i] + hi[i]) / 2), "bandwidth_hz": float(hi[i] - lo[i]), "f_lo": float(lo[i]),
         "f_hi": float(hi[i]), "peak_hz": float(freqs[peak_bin[i]]),
         "peak_db": float(10 * np.log10(power[peak_bin[i]])), "snr_db": float(snr[i])}
        for i in keep
    ]


SNR_MIN_DB, SNR_MAX_DB = 0.0, 60.0
SPECTRAL_MIN_DB = 3.0


def _is_clipped(x, full_scale=0.98, min_fraction=0.01):
    """True when a noticeable share of I or Q samples sit on the ADC rail (normalized full scale is 1)."""
    comp = np.concatenate([np.abs(x.real), np.abs(x.imag)])
    peak = float(np.max(comp))
    return peak >= full_scale and float(np.mean(comp >= 0.999 * peak)) > min_fraction


def estimate_snr_safe(iq, sample_rate, nperseg=1024):
    """SNR in dB that is always a finite float in [0, 60] - never None.

    Tries the spectral energy-separation estimate (estimate_snr) first, which works for
    oversampled, band-limited signals. If the signal fills the band or nothing stands out,
    falls back to the blind M2M4 moment estimator, and if that is invalid too (pure noise, no
    signal, too few samples) or the samples are clipped, returns 0.0. Results below 0 dB are
    reported as 0.0. M2M4 assumes constant-modulus symbols, so it over- or under-reads QAM by a few dB.

    Returns:
        (snr_db, method) where method is 'spectral', 'M2M4', 'clipped' or 'default'.
    """
    x = np.nan_to_num(np.asarray(iq, dtype=np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 100 or not np.any(x):
        return SNR_MIN_DB, "default"
    if _is_clipped(x):
        return SNR_MIN_DB, "clipped"
    seg = min(int(nperseg), x.size // 8)  # average at least ~8 frames or the noise floor is meaningless
    for name, estimator, accept_above in (
        # a spectral result under ~3 dB is what noise alone produces (few averaged frames), so it
        # only counts as a signal above that; otherwise try the moment estimator
        ("spectral", lambda: estimate_snr(x, sample_rate, nperseg=seg), SPECTRAL_MIN_DB),
        ("M2M4", lambda: estimate_snr_m2m4(x), -np.inf),
    ):
        if name == "spectral" and seg < 32:
            continue
        try:
            value = float(estimator())
        except (ValueError, FloatingPointError):
            continue
        if np.isnan(value) or value < accept_above:
            continue
        return float(np.clip(value, SNR_MIN_DB, SNR_MAX_DB)), name
    return SNR_MIN_DB, "default"


def peak_level_db(iq, nperseg=1024):
    """Level of the strongest spectral line in a block, in dB relative to full scale (a full-scale
    complex tone reads 0 dB). This is the peak of compute_stft_row, the quantity the waterfall shows."""
    _, row = compute_stft_row(iq, 1.0, nperseg)
    return float(np.max(row))


class SquelchGate:
    """Squelch with hysteresis and hang time.

    Opens when the level reaches threshold_db. Once open it stays open while the level is within
    hysteresis_db below the threshold, and only closes after the level has been under that
    lower edge for hang_s seconds, so a signal riding on the threshold does not chatter.
    """

    def __init__(self, threshold_db=-50.0, hysteresis_db=3.0, hang_s=0.4):
        self.threshold_db, self.hysteresis_db, self.hang_s = float(threshold_db), float(hysteresis_db), float(hang_s)
        self.is_open = False
        self._last_above = -np.inf

    def reset(self):
        self.is_open = False
        self._last_above = -np.inf

    def update(self, level_db, t_s):
        """Feed one level reading taken at time t_s (seconds). Returns (is_open, changed)."""
        was_open = self.is_open
        if level_db >= self.threshold_db or (was_open and level_db >= self.threshold_db - self.hysteresis_db):
            self.is_open = True
            self._last_above = t_s
        elif was_open and t_s - self._last_above > self.hang_s:
            self.is_open = False
        return self.is_open, self.is_open != was_open
