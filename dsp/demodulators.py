"""Digital demodulators (FSK, PSK, QAM) returning raw bits as uint8 arrays of 0/1.

Conventions
-----------
* FSK: Gray-coded, lowest frequency first: 2-FSK -> 0, 1; 4-FSK -> 00, 01, 11, 10.
* BPSK: +1 -> 0, -1 -> 1.  QPSK (points on the diagonals): bits = [I < 0, Q < 0].
* QAM: square constellation, Gray-coded per axis, bits = [I bits, Q bits].
* Blind PSK/QAM carrier recovery leaves a phase ambiguity (180 deg BPSK, 90 deg
  QPSK/QAM). Pass differential=True if the transmitter used differential encoding.
"""

import numpy as np
from scipy import signal

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
    best = max(offsets, key=lambda o: np.var(_window_means(disc, sps, o)))
    vals = _window_means(disc, sps, best)

    if order == 2:
        return (vals > 0).astype(np.uint8)

    a = np.mean(np.abs(vals)) / 2.0  # levels are -3a, -a, a, 3a
    level = np.digitize(vals, [-2 * a, 0.0, 2 * a])  # 0..3, lowest frequency first
    gray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray[level].ravel()


def _interp_cubic(x, pos):
    i = int(np.floor(pos))
    mu = pos - i
    c = (
        -mu * (mu - 1) * (mu - 2) / 6.0,
        (mu + 1) * (mu - 1) * (mu - 2) / 2.0,
        -(mu + 1) * mu * (mu - 2) / 2.0,
        (mu + 1) * mu * (mu - 1) / 6.0,
    )
    return c[0] * x[i - 1] + c[1] * x[i] + c[2] * x[i + 1] + c[3] * x[i + 2]


def _pi_gains(bandwidth, damping=0.7071):
    theta = bandwidth / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta + theta ** 2
    return 4.0 * damping * theta / denom, 4.0 * theta ** 2 / denom


def _coarse_cfo_correct(iq, power):
    """Remove carrier offset via the power-law spectral line (2 = BPSK, 4 = QPSK/QAM)."""
    n_fft = 1 << int(np.ceil(np.log2(len(iq)))) + 3
    spec = np.abs(np.fft.fft(iq ** power, n_fft))
    freq = np.fft.fftfreq(n_fft)[np.argmax(spec)] / power
    return iq * np.exp(-2j * np.pi * freq * np.arange(len(iq)))


def gardner_timing_recovery(iq, sps, bandwidth=0.02):
    """Gardner timing-error-detector loop; returns one complex sample per symbol.

    The detector is insensitive to carrier phase, so it can run before carrier recovery.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    kp, ki = _pi_gains(bandwidth)
    half = sps / 2.0
    power = np.abs(iq) ** 2
    starts = np.arange(0.0, sps, 0.25)
    metric = [
        np.mean(power[np.round(o + sps * np.arange(int((len(iq) - o - 2) // sps))).astype(int)])
        for o in starts
    ]
    pos = starts[int(np.argmax(metric))] + sps + 1.0
    prev = _interp_cubic(iq, pos)
    out = [prev]
    integ = 0.0
    while pos + sps + 2 < len(iq) - 2:
        pos_next = pos + sps
        cur = _interp_cubic(iq, pos_next)
        mid = _interp_cubic(iq, pos_next - half)
        err = -np.real(np.conj(mid) * (cur - prev))
        err = np.clip(err, -1.0, 1.0)
        integ += ki * err
        pos = pos_next + kp * err + integ
        cur = _interp_cubic(iq, pos)
        out.append(cur)
        prev = cur
    return np.asarray(out)


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
    out = np.empty(len(symbols), dtype=np.complex128)
    phase = 0.0
    freq = 0.0
    for n, y in enumerate(symbols):
        z = y * np.exp(-1j * phase)
        out[n] = z
        if mode == "bpsk":
            err = np.sign(z.real) * z.imag
        elif mode == "qpsk":
            err = (np.sign(z.real) * z.imag - np.sign(z.imag) * z.real) / np.sqrt(2.0)
        else:
            _, _, pts = _slice_qam(np.array([z]), order)
            d = pts[0]
            err = np.imag(z * np.conj(d)) / max(abs(d) ** 2, 1e-6)
            err = np.clip(err, -1.0, 1.0)
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
