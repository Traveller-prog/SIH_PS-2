"""Digital demodulators (FSK, PSK, QAM) returning raw bits as uint8 arrays of 0/1.

Conventions
-----------
* FSK: Gray-coded, lowest frequency first: 2-FSK -> 0, 1; 4-FSK -> 00, 01, 11, 10.
* BPSK: +1 -> 0, -1 -> 1.  QPSK (points on the diagonals): bits = [I < 0, Q < 0].
* QAM: square constellation, Gray-coded per axis, bits = [I bits, Q bits].
* Blind PSK/QAM carrier recovery leaves a phase ambiguity (180 deg BPSK, 90 deg
  QPSK/QAM). Pass differential=True if the transmitter used differential encoding.
"""

import cmath
import math

import numpy as np
from scipy import fft as sfft
from scipy import signal

from dsp.spectral import shift_frequency

SUPPORTED = ("2-FSK", "4-FSK", "BPSK", "QPSK", "16-QAM", "64-QAM")


def rrc_filter(sps, beta=0.35, span=8):
    """Unit-energy root-raised-cosine pulse, `span` symbols each side."""
    half = int(round(span * sps))
    n = np.arange(-half, half + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * n * (1 - beta)) + 4 * beta * n * np.cos(np.pi * n * (1 + beta))
        den = np.pi * n * (1 - (4 * beta * n) ** 2)
        h = num / den
    h[np.isclose(n, 0)] = 1 - beta + 4 * beta / np.pi
    edge = np.isclose(np.abs(n), 1 / (4 * beta))
    h[edge] = (beta / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
    )
    return h / np.sqrt(np.sum(h ** 2))


def _window_means(x, sps, offset):
    """Mean of x over consecutive windows of (fractional) length sps starting at offset."""
    csum = np.concatenate([[0.0], np.cumsum(x)])
    edges = offset + sps * np.arange(int((len(x) - offset) // sps) + 1)
    edges = edges[edges <= len(x)]
    idx = np.clip(np.round(edges).astype(int), 0, len(x))
    lens = np.maximum(np.diff(idx), 1)
    return (csum[idx[1:]] - csum[idx[:-1]]) / lens


def demod_fsk(iq, sample_rate, symbol_rate, order=2):
    """Quadrature-discriminator FSK demodulation with mark/space (level) slicing.

    Args:
        iq: Complex baseband samples.
        sample_rate: Hz.
        symbol_rate: Baud (Hz).
        order: 2 or 4.

    Returns:
        uint8 bit array.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    iq = np.asarray(iq, dtype=np.complex128)
    sps = sample_rate / symbol_rate
    if sps < 2 or len(iq) < 2 * sps:
        raise ValueError("Need at least 2 samples per symbol and 2 symbols of data")

    disc = np.angle(iq[1:] * np.conj(iq[:-1])) * sample_rate / (2 * np.pi)
    disc = disc - np.mean(disc)  # symbols are ~balanced, so the mean is the carrier offset

    # Symbol timing: windows aligned to symbol edges give the most spread-out averages.
    offsets = np.linspace(0, sps, 16, endpoint=False)
    csum = np.concatenate([[0.0], np.cumsum(disc)])  # one cumulative sum shared by every offset
    n_win = int((len(disc) - offsets[-1]) // sps) + 1
    idx = np.clip(np.round(offsets[:, None] + sps * np.arange(n_win)).astype(np.intp), 0, len(disc))
    means = (csum[idx[:, 1:]] - csum[idx[:, :-1]]) / np.maximum(np.diff(idx, axis=1), 1)
    best = offsets[int(np.argmax(means.var(axis=1)))]
    vals = _window_means(disc, sps, best)

    if order == 2:
        return (vals > 0).astype(np.uint8)

    a = np.mean(np.abs(vals)) / 2.0  # levels are -3a, -a, a, 3a
    level = np.digitize(vals, [-2 * a, 0.0, 2 * a])  # 0..3, lowest frequency first
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray[level].ravel()


def _interp_cubic(x, pos):
    """Cubic Lagrange interpolation of x (a list or array) at fractional index pos."""
    i = math.floor(pos)
    mu = pos - i
    return (
        -mu * (mu - 1) * (mu - 2) / 6.0 * x[i - 1]
        + (mu + 1) * (mu - 1) * (mu - 2) / 2.0 * x[i]
        - (mu + 1) * mu * (mu - 2) / 2.0 * x[i + 1]
        + (mu + 1) * mu * (mu - 1) / 6.0 * x[i + 2]
    )


def _pi_gains(bandwidth, damping=0.7071):
    theta = bandwidth / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta + theta ** 2
    return 4.0 * damping * theta / denom, 4.0 * theta ** 2 / denom


def _coarse_cfo_correct(iq, power):
    """Remove carrier offset via the power-law spectral line (2 = BPSK, 4 = QPSK/QAM)."""
    n_fft = 1 << int(np.ceil(np.log2(len(iq)))) + 3
    z = sfft.fft(iq ** power, n_fft, workers=-1)
    freq = sfft.fftfreq(n_fft)[np.argmax(z.real ** 2 + z.imag ** 2)] / power
    return shift_frequency(iq, -freq).astype(np.complex128, copy=False)


def gardner_timing_recovery(iq, sps, bandwidth=0.02):
    """Gardner timing-error-detector loop; returns one complex sample per symbol.

    The detector is insensitive to carrier phase, so it can run before carrier recovery.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    kp, ki = _pi_gains(bandwidth)
    half = sps / 2.0
    power = np.abs(iq) ** 2
    starts = np.arange(0.0, sps, 0.25)
    n_sym = int((len(iq) - starts[-1] - 2) // sps)  # symbols available at every start
    metric = power[np.round(starts[:, None] + sps * np.arange(n_sym)).astype(np.intp)].mean(axis=1)
    pos = float(starts[int(np.argmax(metric))] + sps + 1.0)  # plain floats: NumPy scalars make the loop 2x slower
    sps, kp, ki, half = float(sps), float(kp), float(ki), float(half)

    # The loop is a feedback recursion (each step depends on the last error), so it cannot be
    # vectorized. Python-list samples make the scalar interpolation cheap; the output array grows
    # geometrically instead of a list of NumPy scalars being converted at the end.
    xs = iq.tolist()
    out = np.empty(int(len(iq) / sps) + 16, dtype=np.complex128)
    prev = _interp_cubic(xs, pos)
    out[0] = prev
    count = 1
    integ = 0.0
    while pos + sps + 2 < len(xs) - 2:
        pos_next = pos + sps
        cur = _interp_cubic(xs, pos_next)
        mid = _interp_cubic(xs, pos_next - half)
        err = -(mid.conjugate() * (cur - prev)).real
        err = min(max(err, -1.0), 1.0)
        integ += ki * err
        pos = pos_next + kp * err + integ
        cur = _interp_cubic(xs, pos)
        if count == len(out):
            out = np.concatenate([out, np.empty_like(out)])
        out[count] = cur
        count += 1
        prev = cur
    return out[:count]


def _qam_levels(order):
    k = int(round(np.sqrt(order)))
    return np.arange(-(k - 1), k, 2), k


def _slice_qam(z, order):
    """Nearest-point decision on unit-average-power square QAM; returns (I_idx, Q_idx, points)."""
    levels, k = _qam_levels(order)
    scale = np.sqrt(2.0 * (order - 1) / 3.0)
    ii = np.clip(np.round((z.real * scale + (k - 1)) / 2.0), 0, k - 1).astype(int)
    qq = np.clip(np.round((z.imag * scale + (k - 1)) / 2.0), 0, k - 1).astype(int)
    pts = (levels[ii] + 1j * levels[qq]) / scale
    return ii, qq, pts


def costas_loop(symbols, mode, bandwidth=None, order=16):
    """Symbol-rate Costas loop. mode: 'bpsk', 'qpsk' or 'qam' (decision-directed)."""
    if bandwidth is None:
        bandwidth = 0.01 if mode == "qam" else 0.03
    kp, ki = _pi_gains(bandwidth)
    # Feedback recursion: sequential by nature. Plain-Python scalars (cmath, no per-symbol array
    # creation) keep it fast; the decision-directed QAM slicer is inlined for the same reason.
    out = np.empty(len(symbols), dtype=np.complex128)
    phase = 0.0
    freq = 0.0
    k = int(round(math.sqrt(order)))
    scale = math.sqrt(2.0 * (order - 1) / 3.0)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for n, y in enumerate(np.asarray(symbols, dtype=np.complex128).tolist()):
        z = y * cmath.exp(-1j * phase)
        out[n] = z
        zr, zi = z.real, z.imag
        if mode == "bpsk":
            err = ((zr > 0) - (zr < 0)) * zi
        elif mode == "qpsk":
            err = (((zr > 0) - (zr < 0)) * zi - ((zi > 0) - (zi < 0)) * zr) * inv_sqrt2
        else:
            ii = min(max(round((zr * scale + (k - 1)) / 2.0), 0), k - 1)
            qq = min(max(round((zi * scale + (k - 1)) / 2.0), 0), k - 1)
            d = complex(2 * ii - (k - 1), 2 * qq - (k - 1)) / scale
            err = (z * d.conjugate()).imag / max(abs(d) ** 2, 1e-6)
            err = min(max(err, -1.0), 1.0)
        freq += ki * err
        phase += freq + kp * err
    return out


def recover_symbols(iq, sps, mode, order=16, rolloff=0.35):
    """AGC -> coarse CFO removal -> RRC matched filter -> Gardner timing -> Costas lock.

    rolloff is the transmit RRC roll-off; set it to None to skip matched filtering
    if the input is already matched-filtered.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if sps < 2:
        raise ValueError("sps must be >= 2 for timing recovery")
    if len(iq) < 16 * sps:
        raise ValueError("Not enough samples for synchronization")
    iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2))
    iq = _coarse_cfo_correct(iq, 2 if mode == "bpsk" else 4)
    if rolloff is not None:
        iq = signal.fftconvolve(iq, rrc_filter(sps, rolloff), mode="same")
    syms = gardner_timing_recovery(iq, sps)
    syms = syms / np.sqrt(np.mean(np.abs(syms) ** 2))
    return costas_loop(syms, mode, order=order)


def _differential_decode(bits_per_symbol_matrix):
    """XOR each symbol's bits with the previous symbol's bits."""
    b = bits_per_symbol_matrix
    out = b.copy()
    out[1:] = b[1:] ^ b[:-1]
    return out


def demod_bpsk(iq, sps, differential=False):
    syms = recover_symbols(iq, sps, "bpsk")
    bits = (syms.real < 0).astype(np.uint8)[:, None]
    if differential:
        bits = _differential_decode(bits)
    return bits.ravel()


def demod_qpsk(iq, sps, differential=False):
    syms = recover_symbols(iq, sps, "qpsk")
    bits = np.stack([syms.real < 0, syms.imag < 0], axis=1).astype(np.uint8)
    if differential:
        bits = _differential_decode(bits)
    return bits.ravel()


def slice_qam(symbols, order):
    """Map (already symbol-rate) complex points to bits with a Gray-coded slicer.

    Points are normalized to unit RMS first. Returns a uint8 bit array.
    """
    if order not in (16, 64):
        raise ValueError("order must be 16 or 64")
    z = np.asarray(symbols, dtype=np.complex128)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    ii, qq, _ = _slice_qam(z, order)
    n_bits = int(np.log2(np.sqrt(order)))
    gray = lambda v: v ^ (v >> 1)
    shifts = np.arange(n_bits - 1, -1, -1)
    bi = (gray(ii)[:, None] >> shifts) & 1
    bq = (gray(qq)[:, None] >> shifts) & 1
    return np.concatenate([bi, bq], axis=1).astype(np.uint8).ravel()


def demod_qam(iq, order, sps=1):
    """QAM demodulation. sps=1 means iq is already symbol-rate and synchronized;
    otherwise timing and carrier recovery run first."""
    if sps > 1:
        z = recover_symbols(iq, sps, "qam", order=order)
    else:
        z = np.asarray(iq)
    return slice_qam(z, order)


def demodulate(iq, sample_rate, modulation, symbol_rate, differential=False):
    """Dispatch on a classifier label ('2-FSK', '4-FSK', 'BPSK', 'QPSK', '16-QAM', '64-QAM')."""
    if modulation not in SUPPORTED:
        raise ValueError(f"Unsupported modulation {modulation!r}; choose from {SUPPORTED}")
    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be positive")
    sps = sample_rate / symbol_rate
    if modulation.endswith("FSK"):
        return demod_fsk(iq, sample_rate, symbol_rate, order=int(modulation[0]))
    if modulation == "BPSK":
        return demod_bpsk(iq, sps, differential)
    if modulation == "QPSK":
        return demod_qpsk(iq, sps, differential)
    return demod_qam(iq, int(modulation.split("-")[0]), sps)


# ---- constellation quality metrics ------------------------------------------------------

LINEAR_MODULATIONS = ("BPSK", "QPSK", "16-QAM", "64-QAM")


def ideal_constellation(modulation):
    """Unit-average-power ideal points: BPSK +-1, QPSK (+-1+-j)/sqrt2, square QAM on a grid."""
    if modulation == "BPSK":
        return np.array([1.0, -1.0], dtype=np.complex128)
    if modulation == "QPSK":
        return (np.array([1, 1, -1, -1]) + 1j * np.array([1, -1, 1, -1])) / np.sqrt(2.0)
    if modulation in ("16-QAM", "64-QAM"):
        order = int(modulation.split("-")[0])
        levels, _ = _qam_levels(order)
        scale = np.sqrt(2.0 * (order - 1) / 3.0)
        return ((levels[:, None] + 1j * levels[None, :]) / scale).ravel()
    raise ValueError(f"EVM applies to {LINEAR_MODULATIONS}, not {modulation!r}")


def recover_constellation(iq, sample_rate, modulation, symbol_rate):
    """Synchronized symbol-rate points (unit RMS) for a PSK/QAM signal: AGC, matched filter,
    Gardner timing and Costas carrier lock. Used to measure EVM; needs >= 16 symbols."""
    if modulation not in LINEAR_MODULATIONS:
        raise ValueError(f"EVM applies to {LINEAR_MODULATIONS}, not {modulation!r}")
    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be positive")
    mode = {"BPSK": "bpsk", "QPSK": "qpsk"}.get(modulation, "qam")
    order = int(modulation.split("-")[0]) if mode == "qam" else 16
    return recover_symbols(iq, sample_rate / symbol_rate, mode, order=order)


def constellation_metrics(symbols, modulation, settle=None):
    """EVM and phase statistics of synchronized symbols against the nearest ideal point.

    Symbols are scaled to unit RMS, then each is compared with its nearest ideal point
    (decision-directed, so the 90/180-degree lock ambiguity does not matter).

    Args:
        symbols: complex symbol-rate points after timing and carrier recovery.
        modulation: 'BPSK', 'QPSK', '16-QAM' or '64-QAM'.
        settle: leading symbols to drop while the loops converge (default 20 % of the
            symbols, at most 100).

    Returns:
        dict with
            evm_rms_pct     RMS error vector / RMS reference power, in percent
            evm_db          20*log10(EVM)
            snr_db          -evm_db: the SNR implied by the EVM (all impairments counted as noise)
            phase_jitter_deg  standard deviation of the symbol phase error
            phase_offset_deg  mean phase error (residual rotation left by the carrier loop)
            n_symbols       symbols used
        EVM is under-estimated when SNR is so low that symbols fall in the wrong decision region.
    """
    z = np.asarray(symbols, dtype=np.complex128)
    if settle is None:
        settle = min(len(z) // 5, 100)
    z = z[settle:]
    if len(z) < 16:
        raise ValueError("Need at least 16 symbols to measure EVM")
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    ref = ideal_constellation(modulation)
    nearest = ref[np.argmin(np.abs(z[:, None] - ref[None, :]), axis=1)]
    err = z - nearest
    evm = float(np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(nearest) ** 2)))
    phase = np.degrees(np.angle(z * np.conj(nearest)))
    return {
        "evm_rms_pct": 100.0 * evm,
        "evm_db": 20.0 * np.log10(max(evm, 1e-9)),
        "snr_db": -20.0 * np.log10(max(evm, 1e-9)),
        "phase_jitter_deg": float(np.std(phase)),
        "phase_offset_deg": float(np.mean(phase)),
        "n_symbols": int(len(z)),
    }
