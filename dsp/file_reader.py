"""Readers for raw .IQ recordings and .wav files, returning complex64 arrays."""

import re
import threading
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy import signal
from scipy.io import wavfile

from dsp.spectral import shift_frequency

_RAW_FORMATS = {
    "int16": np.dtype("<i2"),
    "float32": np.dtype("<f4"),
}

_RATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([kKmMgG]?)(?:sps|sa/s|hz)(?![A-Za-z])",
    re.IGNORECASE,
)
_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def detect_sample_rate_from_name(path):
    """Parse a sample rate from a filename such as 'cap_2.4Msps.iq' or 'x_250kHz.iq'.

    Returns the rate in Hz, or None if the name carries no recognisable rate.
    """
    match = _RATE_PATTERN.search(Path(path).stem)
    if not match:
        return None
    return float(match.group(1)) * _SCALE[match.group(2).lower()]


class IQSource:
    """Lazy complex64 view of an IQ recording: slices are loaded on demand.

    Raw .IQ files are opened with numpy.memmap, so nothing is read until a slice or set of
    frames is requested; an in-memory complex array (e.g. from a .wav) can be wrapped too.
    Samples are normalized like read_iq_file (int16 / 32768, float32 as-is).
    """

    def __init__(self, data, sample_rate, scale=1.0):
        self._data = data  # (n, 2) memmap of raw values, or a 1-D complex array
        self._scale = np.float32(scale)
        self.sample_rate = float(sample_rate)
        self._raw = data.ndim == 2

    @classmethod
    def from_array(cls, iq, sample_rate):
        return cls(np.asarray(iq, dtype=np.complex64), sample_rate)

    def __len__(self):
        return int(self._data.shape[0])

    @property
    def is_memmap(self):
        return self._raw

    @property
    def size(self):
        return len(self)

    def _convert(self, chunk):
        if not self._raw:
            return np.asarray(chunk, dtype=np.complex64)
        chunk = np.asarray(chunk)
        out = np.empty(chunk.shape[:-1], dtype=np.complex64)
        out.real = chunk[..., 0] * self._scale
        out.imag = chunk[..., 1] * self._scale
        return out

    def read(self, start=0, count=None):
        """Load samples [start, start+count) (clamped to the file) as a complex64 array."""
        n = len(self)
        start = min(max(int(start), 0), n)
        stop = n if count is None else min(start + max(int(count), 0), n)
        return self._convert(self._data[start:stop])

    def __getitem__(self, key):
        return self._convert(self._data[key])

    def read_frames(self, starts, length):
        """Load len(starts) frames of `length` samples each -> array (len(starts), length)."""
        starts = np.asarray(starts, dtype=np.int64)
        if starts.size and (starts.min() < 0 or starts.max() + length > len(self)):
            raise ValueError("frame extends beyond the end of the file")
        if starts.size > 1:
            step = int(starts[1] - starts[0])
            if step > 0 and np.all(np.diff(starts) == step):
                # evenly spaced frames: a strided view (no index array, no gather copy) - only the
                # float conversion touches the data
                frames = sliding_window_view(self._data, length, axis=0)[starts[0] : starts[-1] + 1 : step]
                if self._raw:
                    frames = frames.transpose(0, 2, 1)  # (frames, length, I/Q)
                return self._convert(frames)
        return self._convert(self._data[starts[:, None] + np.arange(length, dtype=np.int64)])

    def iter_blocks(self, block_samples=1_048_576):
        """Yield consecutive complex64 blocks covering the whole file."""
        for start in range(0, len(self), block_samples):
            yield self.read(start, block_samples)

    def close(self):
        mm = getattr(self._data, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass


def open_iq_memmap(path, dtype="int16", sample_rate=None):
    """Open a raw interleaved I/Q file lazily with numpy.memmap; returns an IQSource.

    Args:
        path: File path.
        dtype: 'int16' (scaled by 1/32768) or 'float32'.
        sample_rate: Hz; parsed from the filename when None.
    """
    key = str(dtype).lower()
    if key not in _RAW_FORMATS:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_RAW_FORMATS)}")
    np_dtype = _RAW_FORMATS[key]

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if sample_rate is None:
        sample_rate = detect_sample_rate_from_name(path)
        if sample_rate is None:
            raise ValueError(
                f"Sample rate not given and could not be detected from {path.name!r}; "
                "pass sample_rate explicitly."
            )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    n = path.stat().st_size // (2 * np_dtype.itemsize)  # whole I/Q pairs only
    if n == 0:
        raw = np.zeros((0, 2), dtype=np_dtype)  # np.memmap refuses empty files
    else:
        raw = np.memmap(path, dtype=np_dtype, mode="r", shape=(n, 2))
    return IQSource(raw, sample_rate, scale=1.0 / 32768.0 if key == "int16" else 1.0)


def read_iq_file(path, dtype="int16", sample_rate=None, offset_samples=0, max_samples=None):
    """Read a binary interleaved I/Q file (I0, Q0, I1, Q1, ...), little-endian.

    The file is memory-mapped, so only the requested slice is read from disk.

    Args:
        path: File path.
        dtype: 'int16' (scaled to [-1, 1) by 1/32768) or 'float32' (used as-is).
        sample_rate: Sampling rate in Hz. If None, it is parsed from the filename.
        offset_samples: Complex samples to skip at the start.
        max_samples: Maximum complex samples to read (None = all).

    Returns:
        (iq, sample_rate): complex64 array and sampling rate in Hz.
    """
    src = open_iq_memmap(path, dtype, sample_rate)
    try:
        return src.read(offset_samples, max_samples), src.sample_rate
    finally:
        src.close()


def read_wav_file(path, sample_rate=None):
    """Read a .wav file with scipy.io.wavfile as I/Q.

    Stereo (or more) files use channel 0 as I and channel 1 as Q. Mono files are
    treated as real signals (Q = 0). Samples are normalized to [-1, 1].

    Args:
        path: File path.
        sample_rate: Optional override in Hz; otherwise the rate in the WAV header.

    Returns:
        (iq, sample_rate): complex64 array and sampling rate in Hz.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    header_rate, data = wavfile.read(path)
    fs = float(sample_rate) if sample_rate is not None else float(header_rate)
    if fs <= 0:
        raise ValueError("sample_rate must be positive")

    if data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = (data.astype(np.float64) / 2147483648.0).astype(np.float32)
    elif data.dtype in (np.float32, np.float64):
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample type: {data.dtype}")

    if data.ndim == 1:
        iq = data.astype(np.complex64)
    else:
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    return iq, fs


def read_signal_file(path, dtype="int16", sample_rate=None, **kwargs):
    """Dispatch on extension: .wav -> read_wav_file, anything else -> read_iq_file."""
    if Path(path).suffix.lower() == ".wav":
        return read_wav_file(path, sample_rate=sample_rate)
    return read_iq_file(path, dtype=dtype, sample_rate=sample_rate, **kwargs)


# ---------------------------------------------------------------------------------------------
# Live SDR input
# ---------------------------------------------------------------------------------------------

LIVE_BACKENDS = ("RTL-SDR (pyrtlsdr)", "SoapySDR", "Simulated")


class RingBuffer:
    """Thread-safe circular buffer holding the most recent complex64 samples."""

    def __init__(self, capacity):
        self._buf = np.zeros(int(capacity), dtype=np.complex64)
        self._total = 0
        self._lock = threading.Lock()

    def __len__(self):
        return min(self._total, len(self._buf))

    @property
    def total_written(self):
        return self._total

    def write(self, chunk):
        chunk = np.asarray(chunk, dtype=np.complex64)
        cap = len(self._buf)
        if chunk.size >= cap:
            chunk = chunk[-cap:]
        with self._lock:
            pos = self._total % cap
            first = min(chunk.size, cap - pos)
            self._buf[pos : pos + first] = chunk[:first]
            self._buf[: chunk.size - first] = chunk[first:]
            self._total += chunk.size

    def snapshot(self, count=None):
        """Latest `count` samples (default: everything held) in time order, as a copy."""
        with self._lock:
            cap, total = len(self._buf), self._total
            held = min(total, cap)
            count = held if count is None else min(int(count), held)
            idx = np.arange(total - count, total) % cap
            return self._buf[idx].copy()


def backend_available(backend):
    """(ok, reason): whether a live backend's Python package can be imported."""
    try:
        if backend == LIVE_BACKENDS[0]:
            import rtlsdr  # noqa: F401
        elif backend == LIVE_BACKENDS[1]:
            import SoapySDR  # noqa: F401
        elif backend != LIVE_BACKENDS[2]:
            return False, f"Unknown backend {backend!r}"
    except ImportError:
        pkg = "pyrtlsdr (pip install pyrtlsdr)" if backend == LIVE_BACKENDS[0] else "SoapySDR (with its Python bindings)"
        return False, f"{pkg} is not installed"
    return True, ""


class SimulatedDevice:
    """Stand-in receiver for demos and tests: noise, two carriers and a slow frequency sweep.

    Streams at the requested sample rate (read() sleeps to keep real-time pace).
    """

    def __init__(self, center_hz, sample_rate, gain_db=None):
        self.center_hz, self.sample_rate = float(center_hz), float(sample_rate)
        self.gain_db = gain_db
        self._n = 0
        self._t_start = time.perf_counter()
        self._rng = np.random.default_rng()

    def set_center(self, hz):
        self.center_hz = float(hz)

    def set_gain(self, db):
        self.gain_db = db

    def read(self, n):
        fs = self.sample_rate
        t = (self._n + np.arange(n)) / fs
        gain = 0.5 if self.gain_db is None or self.gain_db < 0 else 10 ** ((self.gain_db - 30) / 20)
        sweep = 0.25 * fs * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))  # sweeps 0..fs/4
        x = (0.4 * np.exp(2j * np.pi * (0.10 * fs) * t)
             + 0.25 * np.exp(-2j * np.pi * (0.22 * fs) * t)
             + 0.3 * np.exp(1j * (2 * np.pi * np.cumsum(sweep) / fs)))
        x = gain * x + 0.03 * (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n))
        self._n += n
        lag = self._n / fs - (time.perf_counter() - self._t_start)
        if lag > 0:
            time.sleep(min(lag, 1.0))
        return x.astype(np.complex64)

    def close(self):
        pass


class RtlSdrDevice:
    """RTL-SDR through pyrtlsdr. A negative gain means automatic gain control."""

    def __init__(self, center_hz, sample_rate, gain_db=None):
        from rtlsdr import RtlSdr

        self._sdr = RtlSdr()
        self._sdr.sample_rate = float(sample_rate)
        self._sdr.center_freq = float(center_hz)
        self.set_gain(gain_db)
        self.sample_rate = float(self._sdr.sample_rate)
        self.center_hz = float(self._sdr.center_freq)

    def set_center(self, hz):
        self._sdr.center_freq = float(hz)
        self.center_hz = float(hz)

    def set_gain(self, db):
        self._sdr.gain = "auto" if db is None or db < 0 else float(db)

    def read(self, n):
        n = max(1024, int(n) // 1024 * 1024)  # librtlsdr wants multiples of 512
        return np.asarray(self._sdr.read_samples(n), dtype=np.complex64)

    def close(self):
        self._sdr.close()


class SoapyDevice:
    """Any SoapySDR receiver (first one found). A negative gain means automatic gain."""

    def __init__(self, center_hz, sample_rate, gain_db=None):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

        self._rx = SOAPY_SDR_RX
        self._dev = SoapySDR.Device("")
        self._dev.setSampleRate(SOAPY_SDR_RX, 0, float(sample_rate))
        self._dev.setFrequency(SOAPY_SDR_RX, 0, float(center_hz))
        self.set_gain(gain_db)
        self._stream = self._dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self._dev.activateStream(self._stream)
        self.sample_rate = float(self._dev.getSampleRate(SOAPY_SDR_RX, 0))
        self.center_hz = float(self._dev.getFrequency(SOAPY_SDR_RX, 0))

    def set_center(self, hz):
        self._dev.setFrequency(self._rx, 0, float(hz))
        self.center_hz = float(hz)

    def set_gain(self, db):
        auto = db is None or db < 0
        self._dev.setGainMode(self._rx, 0, auto)
        if not auto:
            self._dev.setGain(self._rx, 0, float(db))

    def read(self, n):
        buf = np.empty(int(n), dtype=np.complex64)
        got = 0
        while got < n:
            sr = self._dev.readStream(self._stream, [buf[got:]], n - got, timeoutUs=1_000_000)
            if sr.ret < 0:
                raise OSError(f"SoapySDR readStream error {sr.ret}")
            got += sr.ret
        return buf

    def close(self):
        self._dev.deactivateStream(self._stream)
        self._dev.closeStream(self._stream)


def open_live_device(backend, center_hz, sample_rate, gain_db=None):
    """Open a live receiver; the returned object has read(n) -> complex64, set_center(hz),
    set_gain(db), close(), and the actual sample_rate / center_hz."""
    ok, why = backend_available(backend)
    if not ok:
        raise RuntimeError(why)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    cls = {LIVE_BACKENDS[0]: RtlSdrDevice, LIVE_BACKENDS[1]: SoapyDevice, LIVE_BACKENDS[2]: SimulatedDevice}[backend]
    return cls(center_hz, sample_rate, gain_db)


# ---------------------------------------------------------------------------------------------
# Streaming analysis: PSD, spectrogram and baseband in bounded-memory chunks
# ---------------------------------------------------------------------------------------------

STREAM_BLOCK = 1 << 20  # samples read per chunk (8 MB as complex64)
STREAM_BUDGET = 32_000_000  # samples analysed at most; larger files are covered by evenly spaced blocks


def _block_starts(n, block, budget):
    """Start index of every contiguous block of the file, or an evenly spaced subset when the file
    is larger than the sample budget (whole blocks keep the disk reads sequential)."""
    n_blocks = -(-n // block)
    if budget is None or n <= budget:
        return np.arange(n_blocks, dtype=np.int64) * block
    keep = max(1, int(budget // block))
    return np.unique(np.linspace(0, n_blocks - 1, keep).astype(np.int64)) * block


def stream_psd(source, sample_rate, nperseg=1024, window="hann", db=True, block_samples=STREAM_BLOCK,
               max_samples=STREAM_BUDGET):
    """Welch PSD of a whole (memory-mapped) file computed one chunk at a time.

    Each chunk is read from the map, viewed as 50 %-overlapping frames without copying, windowed
    into one pre-allocated buffer and transformed in place; only a running power sum is kept, so
    memory stays at a few chunks however long the file is. Chunks start on frame boundaries, so
    the frames are exactly those of a single-pass Welch estimate. Files longer than max_samples
    are covered by evenly spaced chunks (max_samples=None analyses everything).

    Returns (freqs_hz, psd) with the same scaling as dsp.spectral.compute_psd.
    """
    n = len(source)
    if n == 0:
        raise ValueError("source is empty")
    nperseg = min(int(nperseg), n)
    hop = max(nperseg // 2, 1)
    per_block = max(int(block_samples) // hop, 1)
    block = per_block * hop
    w = signal.get_window(window, nperseg)
    w32 = w.astype(np.float32)
    buf = np.empty((per_block, nperseg), dtype=np.complex64)
    acc = np.zeros(nperseg)
    frames_total = 0
    for start in _block_starts(n, block, max_samples):
        x = source.read(start, block + nperseg - hop)
        if x.size < nperseg:
            continue
        frames = sliding_window_view(x, nperseg)[::hop][:per_block]  # zero-copy strided frames
        view = buf[: len(frames)]
        np.multiply(frames, w32, out=view)
        spec = sfft.fft(view, axis=1, overwrite_x=True, workers=-1)
        acc += (spec.real ** 2 + spec.imag ** 2).sum(axis=0, dtype=np.float64)
        frames_total += len(frames)
    if frames_total == 0:
        raise ValueError("file is shorter than one FFT frame")
    psd = sfft.fftshift(acc / (frames_total * sample_rate * np.sum(w ** 2)))
    freqs = sfft.fftshift(sfft.fftfreq(nperseg, 1.0 / sample_rate))
    if db:
        psd = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd


def stream_spectrogram(source, sample_rate, nperseg=1024, rows=400, window="hann", db=True,
                       max_samples=STREAM_BUDGET, max_row_samples=1 << 22):
    """Waterfall matrix of a whole file, streamed: the file is cut into `rows` equal time slices
    and each output row is the power averaged over the frames of one slice, read as a single
    contiguous chunk. No full-resolution STFT is ever held, so memory is one slice plus the
    (nperseg x rows) result. If the file exceeds max_samples, each row uses only the start of
    its slice.

    Returns (freqs_hz, times_s, matrix (nperseg, rows)), matrix in dB magnitude like compute_stft.
    """
    n = len(source)
    if n == 0:
        raise ValueError("source is empty")
    nperseg = min(int(nperseg), n)
    hop = max(nperseg // 2, 1)
    rows = max(1, min(int(rows), n // nperseg))
    slice_len = n // rows
    per_row = min(slice_len, max_row_samples)
    if max_samples is not None:
        per_row = min(per_row, max(nperseg, int(max_samples) // rows))
    w = signal.get_window(window, nperseg)
    w32 = w.astype(np.float32)
    batch = 2048
    buf = np.empty((batch, nperseg), dtype=np.complex64)
    power = np.empty((nperseg, rows))
    for r in range(rows):  # one contiguous read per output row
        x = source.read(r * slice_len, per_row)
        frames = sliding_window_view(x, nperseg)[::hop]
        total = np.zeros(nperseg)
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            view = buf[: len(chunk)]
            np.multiply(chunk, w32, out=view)
            spec = sfft.fft(view, axis=1, overwrite_x=True, workers=-1)
            total += (spec.real ** 2 + spec.imag ** 2).sum(axis=0, dtype=np.float64)
        power[:, r] = total / len(frames)
    mags = sfft.fftshift(np.sqrt(power) / np.sum(w), axes=0)
    freqs = sfft.fftshift(sfft.fftfreq(nperseg, 1.0 / sample_rate))
    times = (np.arange(rows) * slice_len + slice_len / 2) / sample_rate
    if db:
        mags = 20.0 * np.log10(np.maximum(mags, 1e-20))
    return freqs, times, mags


class BasebandStream:
    """Band [f_lo, f_hi] of a file as complex baseband, produced chunk by chunk.

    Each chunk is read from the map, mixed to 0 Hz with a phase that continues across chunks,
    low-pass filtered by overlap-save (the filter keeps its last taps-1 input samples) and
    decimated on a global sample grid, so the concatenated output equals a single-pass
    dsp.spectral.extract_band of the same samples. Iterate to get complex64 blocks; only one
    chunk and the (small) decimated output ever exist at once.

    Attributes: sample_rate (output rate, Hz), decimation.
    """

    def __init__(self, source, sample_rate, f_lo, f_hi, start=0, count=None, chunk_samples=STREAM_BLOCK,
                 oversample=6.0, min_out=1024, max_taps=4095):
        n = len(source)
        self.source, self.fs = source, float(sample_rate)
        self.start = min(max(int(start), 0), n)
        self.count = n - self.start if count is None else min(int(count), n - self.start)
        self.chunk = int(chunk_samples)
        lo, hi = max(float(f_lo), -self.fs / 2), min(float(f_hi), self.fs / 2)
        if hi <= lo:
            raise ValueError("Empty frequency range")
        bw, self.fc = hi - lo, (hi + lo) / 2
        self.passthrough = bw >= 0.9 * self.fs and abs(self.fc) < 0.05 * self.fs
        if self.passthrough:
            self.decimation, self.sample_rate, self.h = 1, self.fs, None
            return
        if self.count < 64:
            raise ValueError("Too few samples in the selected time range")
        taps = int(np.clip(6 * self.fs / bw, 31, max_taps)) | 1
        self.h = signal.firwin(taps, min(bw / 2, 0.49 * self.fs), fs=self.fs)
        self.delay = (taps - 1) // 2
        self.decimation = int(max(1, min(self.fs / (oversample * bw), self.count // min_out)))
        self.sample_rate = self.fs / self.decimation

    def __iter__(self):
        if self.passthrough:
            for off in range(0, self.count, self.chunk):
                yield self.source.read(self.start + off, min(self.chunk, self.count - off))
            return
        taps, delay, decim = len(self.h), self.delay, self.decimation
        tail = np.zeros(taps - 1, dtype=np.complex64)
        w_cyc = -self.fc / self.fs  # cycles per sample of the mixer
        g0 = 0  # global index of the chunk's first sample (relative to `start`)
        total = self.count + delay  # after the last real sample, `delay` zeros flush the filter
        while g0 < total:
            take = min(self.chunk, self.count - g0)
            x = self.source.read(self.start + g0, take) if take > 0 else np.zeros(0, dtype=np.complex64)
            if g0 + x.size >= self.count:  # final chunk: append the flush zeros
                x = np.concatenate([x, np.zeros(total - g0 - x.size, dtype=np.complex64)])
            x = shift_frequency(x, w_cyc) * np.complex64(np.exp(2j * np.pi * ((w_cyc * g0) % 1.0)))
            buf = np.concatenate([tail, x])
            tail = buf[-(taps - 1) :].copy()
            causal = signal.oaconvolve(buf, self.h, mode="valid")  # output for each new sample
            # keep global times t = delay + j*decim (the centred alignment of extract_band)
            first = delay - g0 if g0 <= delay else (delay - g0) % decim
            if first < causal.size:
                yield causal[first :: decim].astype(np.complex64, copy=False)
            g0 += x.size


def stream_baseband(source, sample_rate, f_lo, f_hi, start=0, count=None, chunk_samples=STREAM_BLOCK, **kwargs):
    """Band-limited baseband of source[start:start+count] as (samples, sample_rate, decimation),
    built from a BasebandStream so the raw band never has to be in memory at once."""
    stream = BasebandStream(source, sample_rate, f_lo, f_hi, start, count, chunk_samples, **kwargs)
    blocks = list(stream)
    out = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.complex64)
    return out, stream.sample_rate, stream.decimation


def recording_name(center_hz, sample_rate, stamp=None):
    """File name for a live recording, e.g. live_433p9200MHz_2.4Msps_20260925_141200.iq.

    The sample rate is written as '<n>Msps' / '<n>ksps' so detect_sample_rate_from_name() finds it
    when the file is opened again; the centre frequency uses 'p' for the decimal point so it is
    not mistaken for a sample rate.
    """
    from datetime import datetime

    fs = float(sample_rate)
    rate = f"{fs / 1e6:g}Msps" if fs >= 1e6 else f"{fs / 1e3:g}ksps" if fs >= 1e3 else f"{fs:g}sps"
    centre = f"{center_hz / 1e6:.4f}".replace(".", "p")
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"live_{centre}MHz_{rate}_{stamp}.iq"


class IQRecorder:
    """Appends complex64 buffers to a raw .iq file: interleaved little-endian float32 I, Q pairs
    (open it with the float32 format). Writes are buffered by the OS; bytes_written can be read
    from another thread for a live size counter."""

    def __init__(self, directory, center_hz, sample_rate):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / recording_name(center_hz, sample_rate)
        self._fh = open(self.path, "xb", buffering=1 << 20)  # never overwrite an existing recording
        self.bytes_written = 0
        self.sample_rate = float(sample_rate)

    def write(self, buf):
        data = np.ascontiguousarray(buf, dtype=np.complex64).view(np.float32)
        self._fh.write(data.data)
        self.bytes_written += data.nbytes

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
