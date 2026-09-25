"""Readers for raw .IQ recordings and .wav files, returning complex64 arrays."""

import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile

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
