"""Main window: file sidebar + PSD, waterfall and constellation plots."""

import csv
import inspect
import json
import threading
import time
from fractions import Fraction
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from scipy import signal as sp_signal
from PyQt6 import QtCore, QtGui, QtWidgets

from dsp.file_reader import (
    LIVE_BACKENDS,
    IQRecorder,
    IQSource,
    RingBuffer,
    backend_available,
    detect_sample_rate_from_name,
    open_iq_memmap,
    open_live_device,
    stream_baseband,
    stream_psd,
    stream_spectrogram,
    STREAM_BUDGET,
    read_wav_file,
)
from dsp.spectral import (
    compute_constellation,
    compute_psd,
    compute_stft_row,
    correct_cfo,
    SquelchGate,
    detect_channels,
    extract_band,
    estimate_snr_safe,
    estimate_symbol_rate,
)
from dsp.audio import AUDIO_MODES, demodulate_audio, to_pcm16
from dsp.perf import CpuMeter, format_bytes, process_rss_bytes
from dsp.correlation import (
    PAYLOAD_MODES,
    find_sync,
    parse_frames,
    parse_sync_word,
    payload_to_ascii,
    render_payload,
)
from fec.descrambler import SCHEMES as DESCRAMBLE_SCHEMES
from fec.descrambler import descramble
from dsp.demodulators import (
    LINEAR_MODULATIONS,
    SUPPORTED,
    constellation_metrics,
    demodulate,
    ideal_constellation,
    recover_constellation,
)
from fec.deinterleave import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
)
from fec.decoders import rs_decode, viterbi_decode
from ml.model import FRAME_LEN, ModulationClassifier
from utils.pdf_exporter import build_report

try:  # audio playback is optional: the analyzer still runs without QtMultimedia
    from PyQt6 import QtMultimedia
except ImportError:  # pragma: no cover
    QtMultimedia = None

# Files are memory-mapped; each task loads only the slice it needs.
WINDOW_SAMPLES = 1_048_576  # head of the file: symbol-rate estimation and demodulation
CHANNEL_SCAN_SAMPLES = 131_072  # centre of the file used to identify each detected channel
MAX_SCATTER_POINTS = 2000  # constellation points drawn at once (stride-sliced) to keep updates at 60 FPS
MAX_CHANNELS = 12
LIVE_BITS_INTERVAL_S = 1.0  # while the squelch is open, re-extract bits this often
LIVE_BITS_MIN_SAMPLES = 8192  # smallest gated burst worth demodulating
LIVE_BITS_MAX_SAMPLES = 262_144  # newest gated samples used per extraction
SQUELCH_DEFAULT_DB = -40
RECORDINGS_DIR = ROOT / "recordings"  # live recordings are appended here as .iq files
LIVE_ROWS = 200  # waterfall rows kept while streaming
LIVE_CAPTURE_SECONDS = 4  # recent IQ kept for analysis when the stream stops
LIVE_CAPTURE_MAX = 16_000_000
AUDIO_MAX_SECONDS = 10
AUDIO_MAX_SAMPLES = 8_000_000
TARGET_SPS = 8  # samples/symbol the classifier handles best
ROI_MAX_SAMPLES = 262_144  # longest ROI time slice read from the file
CENTER_SAMPLES = 65_536  # centre of the file: classifier, SNR, CFO, constellation
WATERFALL_FRAMES = 400  # rows in the waterfall (each averages one time slice of the file)
DEMOD_MAX_SAMPLES = 200_000
VIEW_MAX_BYTES = 4096

# name -> (kind, label A, label B, default A, default B)
INTERLEAVERS = {
    "None": None,
    "Block (matrix transpose)": ("block", "Rows", "Columns", 8, 16),
    "Convolutional (shift register)": ("conv", "Branches", "Delay (M)", 12, 17),
    "Diagonal": ("diag", "Rows", "Columns", 8, 16),
}

FEC_DECODERS = {
    "None": None,
    "Viterbi K=7 r1/2 (171, 133)": ("viterbi", 7, (0o171, 0o133)),
    "Viterbi K=3 r1/2 (7, 5)": ("viterbi", 3, (0o7, 0o5)),
    "Viterbi K=9 r1/2 (753, 561)": ("viterbi", 9, (0o753, 0o561)),
    "Reed-Solomon (255, 223)": ("rs", 255, 223),
    "Reed-Solomon (255, 239)": ("rs", 255, 239),
    "Reed-Solomon (64, 48)": ("rs", 64, 48),
}


def run_baseband(iq, fs, scheme, symbol_rate, interleaver, fec, skip_bits=0, invert=False):
    """Demodulator -> de-interleaver -> FEC decoder.

    Returns (stages, notes): stages is an ordered dict of stage name -> uint8 bits.
    """
    stages = {}
    notes = []
    bits = demodulate(iq, fs, scheme, symbol_rate)
    if skip_bits:
        bits = bits[skip_bits:]
    if invert:
        bits = bits ^ 1
    stages["Demodulated"] = bits

    if interleaver is not None:
        kind, a, b = interleaver
        if kind == "block":
            bits = deinterleave_block(bits, a, b)
        elif kind == "diag":
            bits = deinterleave_diagonal(bits, a, b)
        else:
            bits = deinterleave_convolutional(bits, a, b)
        stages["De-interleaved"] = bits

    if fec is not None:
        if fec[0] == "viterbi":
            bits = viterbi_decode(bits, fec[1], fec[2])
        else:
            bits, info = rs_decode(bits, fec[1], fec[2], return_info=True)
            fixed = sum(info["corrected"])
            bad = sum(info["failed"])
            notes.append(
                f"RS: {len(info['failed'])} codewords, {fixed} symbol errors corrected, {bad} uncorrectable"
            )
        stages["FEC decoded"] = bits
    return stages, notes


def format_binary(bits, max_bytes=VIEW_MAX_BYTES):
    """Bits grouped as bytes, 8 bytes per line, with hex byte offsets."""
    bits = np.asarray(bits, dtype=np.uint8)[: max_bytes * 8]
    chars = np.where(bits == 1, "1", "0")
    groups = ["".join(chars[i : i + 8]) for i in range(0, len(chars), 8)]
    lines = [f"{i * 8:06X}  " + " ".join(groups[i : i + 8]) for i in range(0, len(groups), 8)]
    return "\n".join(lines)


def format_hex(bits, max_bytes=VIEW_MAX_BYTES):
    """Bytes (MSB first, last byte zero padded), 16 per line, with hex byte offsets."""
    data = np.packbits(np.asarray(bits, dtype=np.uint8)[: max_bytes * 8])
    lines = [
        f"{i:06X}  " + " ".join(f"{b:02X}" for b in data[i : i + 16]) for i in range(0, len(data), 16)
    ]
    return "\n".join(lines)


CFO_ORDER = {"BPSK": 2, "QPSK": 4, "8-PSK": 8, "16-QAM": 4, "64-QAM": 4}


def derotate(iq, fs, label):
    """M-th power CFO correction for PSK/QAM labels. Returns (iq, cfo_hz); cfo is None if not applicable."""
    order = CFO_ORDER.get(label)
    if order is None:
        return iq, None
    try:
        return correct_cfo(iq, fs, order)
    except ValueError:
        return iq, None


# Starting points, not verified standard parameters: the app has no GMSK demodulator (GMSK is
# read as 2-FSK), and the interleaver sizes are assumptions to adjust for the actual system.
PRESETS = {
    "Custom": None,
    "Marine AIS (GMSK / 9600 Baud)": {
        "demod": "2-FSK", "baud": "9600", "interleaver": "None", "il": None, "sync": "0x7E",
        "note": "GMSK read as 2-FSK. 0x7E is the HDLC flag; NRZI and bit-stuffing are not undone, so "
                "search the Invert option / raw bits. AIS has no interleaving.",
    },
    "APCO P25 (C4FM / 4800 Baud)": {
        "demod": "4-FSK", "baud": "4800", "interleaver": "Block (matrix transpose)", "il": (14, 14),
        "sync": "0x5575F5FF77FF",
        "note": "C4FM read as 4-FSK; the dibit-to-symbol mapping of P25 differs from this app's, so bits "
                "may need remapping. 14x14 block (196 bit) is an approximation of the P25 data interleave.",
    },
    "STANAG HF Telemetry (2-FSK / 75 Baud)": {
        "demod": "2-FSK", "baud": "75", "interleaver": "Block (matrix transpose)", "il": (8, 16),
        "sync": "0xEB90",
        "note": "0xEB90 and the 8x16 interleaver are generic assumptions, not a fixed STANAG value.",
    },
}

DEFAULT_SYNC = "0x1ACFFC1D, 0xEB90"
MAX_FRAMES_SHOWN = 200
FRAME_FIELDS = ("length", "frame_id", "address")


def correlate_frames(bits, markers_text, max_errors=0, descramble_scheme=None, invert=False, scope="frame"):
    """Search bits for each comma/space separated hex marker; parse frames for the best one.

    descramble_scheme / invert / scope are passed on to dsp.correlation.parse_frames; with
    scope='stream' the whole stream is descrambled before the marker search.

    Returns (marker_text, frames). marker_text is None when nothing matched.
    Raises ValueError on an unparsable marker.
    """
    post = {"descramble_scheme": descramble_scheme, "invert": invert, "scope": scope}
    if descramble_scheme and scope == "stream":
        bits = descramble(np.asarray(bits, dtype=np.uint8) ^ (1 if invert else 0), descramble_scheme)
        post = {}
    tokens = [t for t in markers_text.replace(",", " ").replace(";", " ").split() if t]
    if not tokens:
        raise ValueError("Enter at least one hex sync marker (e.g. 0x1ACFFC1D).")
    best, best_hits = None, 0
    for tok in tokens:
        try:
            parse_sync_word(tok)
        except ValueError:
            raise ValueError(f"Invalid sync marker {tok!r}; use hex like 0xEB90.")
        hits = len(find_sync(bits, tok, max_errors=max_errors))
        if hits > best_hits:
            best, best_hits = tok, hits
    if best is None:
        return None, []
    return best, parse_frames(bits, best, max_errors=max_errors, **post)


def frames_to_records(frames):
    """Frame objects -> plain dicts (JSON/CSV friendly)."""
    out = []
    for n, f in enumerate(frames):
        out.append(
            {
                "frame": n,
                "bit_offset": f.start,
                "sync_errors": f.sync.errors,
                "inverted": f.sync.inverted,
                "complete": f.complete,
                "header": dict(f.header),
                "payload_length": len(f.payload),
                "payload_ascii": payload_to_ascii(f.payload),
                "payload_utf8": render_payload(f.payload, "UTF-8"),
                "payload_hex": f.payload.hex(" ").upper(),
                "notes": list(f.notes),
            }
        )
    return out


def write_results(base_path, summary):
    """Write summary to <base>.json and <base>.csv (one CSV row per frame). Returns both paths."""
    base = Path(base_path)
    json_path, csv_path = base.with_suffix(".json"), base.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top = {
        k: summary[k]
        for k in ("file", "timestamp", "modulation", "confidence_percent", "demodulator", "sample_rate_hz",
                  "baud_rate", "snr_db", "evm_rms_pct", "evm_snr_db", "phase_noise_deg", "phase_offset_deg",
                  "sync_marker", "stage", "descrambler", "bits_inverted")
    }
    frame_cols = ["frame", "bit_offset", "sync_errors", "inverted", "complete", *FRAME_FIELDS,
                  "payload_length", "payload_ascii", "payload_hex"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([*top, *frame_cols])
        rows = summary["frames"] or [None]
        for rec in rows:
            if rec is None:
                w.writerow([*top.values(), *[""] * len(frame_cols)])
                continue
            flat = {**rec, **{k: rec["header"].get(k, "") for k in FRAME_FIELDS}}
            w.writerow([*top.values(), *[flat[c] for c in frame_cols]])
    return json_path, csv_path


BG = "#0d1117"  # deep charcoal
PANEL = "#141a22"
RAISED = "#1b232e"
BORDER = "#263140"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
ACCENT = "#22d3ee"  # cyan
ACCENT2 = "#3b82f6"  # blue
GREEN = "#10b981"  # emerald: locks / good
AMBER = "#f59e0b"
RED = "#ef4444"
TRACE = "#22d3ee"

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 4px 6px; }}
#sidebar {{ background: {PANEL}; }}
#title {{ font-size: 18px; font-weight: 700; color: {ACCENT}; }}
#metricValue {{ font-size: 16px; font-weight: 600; }}
#ledText {{ font-weight: 600; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; spacing: 8px; padding: 6px 12px; }}
QToolBar QLabel {{ color: {MUTED}; padding-left: 6px; }}
QPushButton#rec {{ background: {RAISED}; border-color: {RED}; color: {RED}; min-width: 130px; }}
QPushButton#rec:hover {{ background: #3a1a1e; color: white; }}
QPushButton#rec:checked {{ background: {RED}; border-color: {RED}; color: white; }}
QLabel#recIndicator {{ font-weight: 800; font-size: 14px; padding-right: 10px; }}
QPushButton#live {{ background: {GREEN}; border-color: {GREEN}; color: #03241a; min-width: 150px; }}
QPushButton#live:hover {{ background: #34d399; border-color: #34d399; color: #03241a; }}
QPushButton#live:checked {{ background: {RED}; border-color: {RED}; color: white; }}
QPushButton#live:checked:hover {{ background: #f87171; border-color: #f87171; color: white; }}
#metricsBar {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 12px;
    font-size: 14px; font-weight: 600; }}

QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px;
    color: {ACCENT}; font-weight: 700; font-size: 11px; }}

QDoubleSpinBox {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 8px; }}
QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QLineEdit, QComboBox, QSpinBox {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 8px;
    selection-background-color: {ACCENT2}; }}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: #3a4a60; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: #4b5666; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: none; }}
QComboBox::down-arrow {{ width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid {ACCENT}; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {ACCENT}; border-radius: 6px;
    selection-background-color: {ACCENT2}; selection-color: white; outline: none; padding: 2px; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 14px; font-weight: 600; }}
QPushButton:hover {{ background: #243042; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: #0f1620; }}
QPushButton:disabled {{ color: #4b5666; border-color: #1c2532; background: {PANEL}; }}
QPushButton#primary {{ background: {ACCENT2}; border-color: {ACCENT2}; color: white; }}
QPushButton#primary:hover {{ background: {ACCENT}; border-color: {ACCENT}; color: #04222a; }}
QPushButton#primary:pressed {{ background: #0e7490; border-color: #0e7490; color: white; }}
QPushButton#primary:disabled {{ background: #182a48; border-color: #182a48; color: #5c6f8f; }}
QPushButton#success {{ background: {GREEN}; border-color: {GREEN}; color: #03241a; }}
QPushButton#success:hover {{ background: #34d399; border-color: #34d399; color: #03241a; }}
QPushButton#success:disabled {{ background: #0f3a2e; border-color: #0f3a2e; color: #3f7a67; }}

QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3a4a60; background: {BG}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 6px; background: {BG}; border: 1px solid {BORDER}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT2}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: white; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2c394b; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:horizontal {{ background: #2c394b; border-radius: 4px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {ACCENT2}; }}
QScrollBar::handle:pressed {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 12px 30px; font-size: 14px; font-weight: 600; border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {RAISED}; color: {ACCENT}; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QMenuBar {{ background: {PANEL}; color: {TEXT}; }}
QMenuBar::item {{ padding: 5px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {RAISED}; color: {ACCENT}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT2}; color: white; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QPlainTextEdit {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px;
    font-family: Consolas, "Courier New", monospace; font-size: 12px; selection-background-color: {ACCENT2}; }}
"""


class Led(QtWidgets.QWidget):
    """Round status LED with a soft glow."""

    def __init__(self, color=MUTED, size=16):
        super().__init__()
        self.setFixedSize(size, size)
        self._color = QtGui.QColor(color)

    def set_color(self, color):
        self._color = QtGui.QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        c = QtCore.QPointF(self.rect().center())
        r = self.width() / 2.0
        glow = QtGui.QRadialGradient(c, r)
        halo = QtGui.QColor(self._color)
        halo.setAlpha(90)
        clear = QtGui.QColor(self._color)
        clear.setAlpha(0)
        glow.setColorAt(0.55, halo)
        glow.setColorAt(1.0, clear)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, r, r)
        core = QtGui.QRadialGradient(QtCore.QPointF(c.x() - r * 0.15, c.y() - r * 0.2), r * 0.6)
        core.setColorAt(0.0, self._color.lighter(150))
        core.setColorAt(1.0, self._color)
        p.setBrush(core)
        p.drawEllipse(c, r * 0.5, r * 0.5)


class QualityBar(QtWidgets.QWidget):
    """SNR quality bar that animates to its new level; colour follows the quality band."""

    MAX_DB = 30.0
    BANDS = ((6.0, RED), (12.0, AMBER), (20.0, ACCENT), (1e9, GREEN))
    NAMES = ("Poor", "Fair", "Good", "Excellent")

    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)
        self._level = 0.0
        self._anim = QtCore.QPropertyAnimation(self, b"level", self)
        self._anim.setDuration(600)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

    def _get_level(self):
        return self._level

    def _set_level(self, value):
        self._level = float(value)
        self.update()

    level = QtCore.pyqtProperty(float, _get_level, _set_level)

    def set_snr(self, snr_db):
        """Animate to the level for snr_db (None -> empty)."""
        target = 0.0 if snr_db is None else min(max(float(snr_db) / self.MAX_DB, 0.0), 1.0)
        self._anim.stop()
        self._anim.setStartValue(self._level)
        self._anim.setEndValue(target)
        self._anim.start()

    @classmethod
    def quality(cls, snr_db):
        return next(name for (limit, _), name in zip(cls.BANDS, cls.NAMES) if snr_db < limit)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        p.setPen(QtGui.QPen(QtGui.QColor(BORDER)))
        p.setBrush(QtGui.QColor(BG))
        p.drawRoundedRect(rect, radius, radius)
        if self._level > 0.005:
            fill = QtCore.QRectF(rect)
            fill.setWidth(max(rect.height(), rect.width() * self._level))
            db = self._level * self.MAX_DB
            color = QtGui.QColor(next(c for limit, c in self.BANDS if db < limit))
            grad = QtGui.QLinearGradient(fill.topLeft(), fill.topRight())
            grad.setColorAt(0.0, color.darker(140))
            grad.setColorAt(1.0, color)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(grad)
            p.drawRoundedRect(fill, radius, radius)


# Spectrum values are relative to full scale unless the receiver's calibration is added here.
DBM_OFFSET_DB = 0.0

SNR_METHOD_NOTES = {
    "spectral": "Spectral energy separation: signal band power against the noise floor outside it",
    "M2M4": "M2M4 moment estimator (the signal fills the band, so no spectral noise reference)",
    "clipped": "Samples are clipped at full scale, so no reliable estimate: reported as 0.0 dB",
    "default": "No signal found (noise only, silence or too few samples): reported as 0.0 dB",
}


def fmt_freq(hz):
    """Signed frequency as Hz / kHz / MHz."""
    a = abs(hz)
    if a >= 1e6:
        return f"{hz / 1e6:.4f} MHz"
    if a >= 1e3:
        return f"{hz / 1e3:.3f} kHz"
    return f"{hz:.1f} Hz"


def fmt_time(sec):
    return f"{sec * 1e3:.3f} ms" if abs(sec) < 1 else f"{sec:.4f} s"


class LiveWorker(QtCore.QThread):
    """Reads IQ buffers from a live receiver, keeps every buffer in a ring buffer and hands
    the GUI the newest one (buffers arriving while the GUI is still busy are dropped)."""

    chunk = QtCore.pyqtSignal(object)
    opened = QtCore.pyqtSignal(float, float)  # actual sample rate, centre frequency
    failed = QtCore.pyqtSignal(str)
    record_started = QtCore.pyqtSignal(str)  # file path
    record_stopped = QtCore.pyqtSignal(str, int)  # file path, bytes written
    record_failed = QtCore.pyqtSignal(str)

    def __init__(self, backend, center_hz, sample_rate, gain_db, chunk_samples, ring, record=False,
                 record_dir=None):
        super().__init__()
        self.record_dir = record_dir
        self.record_on_start = record
        self.recorder = None
        self.recordings = []  # (path, bytes) of every finished recording
        self.args = (backend, center_hz, sample_rate, gain_db)
        self.chunk_samples = int(chunk_samples)
        self.ring = ring
        self.gui_ready = True
        self.dropped = 0
        self._stop_flag = False
        self._pending = {}
        self._lock = threading.Lock()

    def request(self, **changes):
        with self._lock:
            self._pending.update(changes)

    def stop(self):
        self._stop_flag = True

    @property
    def rec_bytes(self):
        rec = self.recorder
        return rec.bytes_written if rec is not None else 0

    def _start_recording(self, device):
        if self.recorder is not None:
            return
        try:
            self.recorder = IQRecorder(self.record_dir, device.center_hz, device.sample_rate)
        except OSError as exc:
            self.record_failed.emit(f"Cannot start recording: {exc}")
            return
        self.record_started.emit(str(self.recorder.path))

    def _stop_recording(self):
        rec, self.recorder = self.recorder, None
        if rec is None:
            return
        rec.close()
        self.recordings.append((str(rec.path), rec.bytes_written))
        self.record_stopped.emit(str(rec.path), rec.bytes_written)

    def run(self):
        try:
            device = open_live_device(*self.args)
        except Exception as exc:  # missing package, no hardware, bad rate...
            self.failed.emit(str(exc))
            return
        self.opened.emit(device.sample_rate, device.center_hz)
        if self.record_on_start:
            self._start_recording(device)
        try:
            while not self._stop_flag:
                with self._lock:
                    changes, self._pending = self._pending, {}
                if "center_hz" in changes:
                    device.set_center(changes["center_hz"])
                if "gain_db" in changes:
                    device.set_gain(changes["gain_db"])
                if "record" in changes:
                    self._start_recording(device) if changes["record"] else self._stop_recording()
                buf = device.read(self.chunk_samples)
                self.ring.write(buf)
                if self.recorder is not None:
                    try:
                        self.recorder.write(buf)
                    except OSError as exc:  # disk full, drive removed...
                        self._stop_recording()
                        self.record_failed.emit(f"Recording stopped: {exc}")
                if self.gui_ready:
                    self.gui_ready = False
                    self.chunk.emit(buf)
                else:
                    self.dropped += 1
        except Exception as exc:
            self.failed.emit(f"Stream error: {exc}")
        finally:
            self._stop_recording()
            device.close()


class Crosshair(QtCore.QObject):
    """Vertical/horizontal hover crosshair with a live read-out on a pyqtgraph PlotItem.

    describe(x, y) -> (text, snapped_y) or None; snapped_y (if not None) is where the
    horizontal line is drawn, otherwise it follows the mouse.
    """

    def __init__(self, plot_item, view_widget, describe):
        super().__init__(view_widget)
        self.plot, self.describe = plot_item, describe
        pen = pg.mkPen(ACCENT, width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.label = pg.TextItem(color=TEXT, fill=pg.mkBrush(13, 17, 23, 225), border=pg.mkPen(ACCENT), anchor=(0, 1))
        for item in (self.vline, self.hline, self.label):
            item.setZValue(1000)
            plot_item.addItem(item, ignoreBounds=True)
        self.hide()
        self.proxy = pg.SignalProxy(plot_item.scene().sigMouseMoved, rateLimit=60, slot=self._moved)
        view_widget.viewport().installEventFilter(self)

    def hide(self):
        for item in (self.vline, self.hline, self.label):
            item.hide()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hide()
        return False

    def _moved(self, args):
        pos = args[0]
        vb = self.plot.vb
        if not vb.sceneBoundingRect().contains(pos):
            self.hide()
            return
        pt = vb.mapSceneToView(pos)
        info = self.describe(pt.x(), pt.y())
        if info is None:
            self.hide()
            return
        text, snap_y = info
        y = pt.y() if snap_y is None else snap_y
        self.vline.setPos(pt.x())
        self.hline.setPos(y)
        self.label.setText(text)
        (x0, x1), (y0, y1) = vb.viewRange()
        self.label.setAnchor((1 if pt.x() > (x0 + x1) / 2 else 0, 0 if y < (y0 + y1) / 2 else 1))
        self.label.setPos(pt.x(), y)
        for item in (self.vline, self.hline, self.label):
            item.show()


def parse_rate(text):
    """Parse '2.4e6', '2400000', '2.4M', '250k' into Hz."""
    text = text.strip().lower().replace("hz", "").replace("sps", "").strip()
    scale = 1.0
    if text and text[-1] in "kmg":
        scale = {"k": 1e3, "m": 1e6, "g": 1e9}[text[-1]]
        text = text[:-1]
    rate = float(text) * scale
    if rate <= 0:
        raise ValueError("Sample rate must be positive")
    return rate


def style_plot(plot, title, x_label, y_label):
    plot.setTitle(title, color="#c9cdd8", size="11pt")
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    plot.showGrid(x=True, y=True, alpha=0.15)


# ---------------------------------------------------------------------------------------------
# Background work. Everything below runs on a Task (QThread) and touches no widget: it receives
# plain data (arrays, dicts, the read-only classifier) and hands results back through signals.
# ---------------------------------------------------------------------------------------------


class Task(QtCore.QThread):
    """Runs fn(report, *args) off the GUI thread.

    fn calls report(kind, payload) to publish partial results and returns the final result.
    All three signals carry the task itself first, so the receiver is a bound method of the
    main window (queued into the GUI thread) that looks up the handlers stored on the task.
    """

    progress = QtCore.pyqtSignal(object, str, object)  # task, kind, payload
    done = QtCore.pyqtSignal(object, object)  # task, result
    failed = QtCore.pyqtSignal(object, object)  # task, exception

    def __init__(self, name, fn, args=(), handlers=None, on_done=None, on_fail=None, parent=None):
        super().__init__(parent)
        self.name, self._fn, self._args = name, fn, args
        self.handlers, self.on_done, self.on_fail = handlers or {}, on_done, on_fail
        self.elapsed = 0.0
        self.seq = 0

    def run(self):
        t0 = time.perf_counter()
        try:
            result = self._fn(lambda kind, payload=None: self.progress.emit(self, kind, payload), *self._args)
        except Exception as exc:  # reported to the GUI thread, never swallowed
            self.elapsed = time.perf_counter() - t0
            self.failed.emit(self, exc)
            return
        self.elapsed = time.perf_counter() - t0
        self.done.emit(self, result)


def band_signal(raw, fs, f_lo, f_hi, baud_hint=None):
    """Isolate [f_lo, f_hi] from raw samples and, since the classifier was trained on 4-10
    samples per symbol, resample to 8 when the symbol rate is known or can be estimated.

    Returns (samples, sample_rate, decimation, note)."""
    y, fs_out, decim = extract_band(raw, fs, f_lo, f_hi)
    resampled = ""
    try:
        baud = baud_hint or estimate_symbol_rate(y, fs_out)[0]
    except ValueError:
        baud = None
    if baud and not 3.5 <= fs_out / baud <= 11.0:
        ratio = Fraction(TARGET_SPS * baud / fs_out).limit_denominator(100)
        y = sp_signal.resample_poly(y, ratio.numerator, ratio.denominator).astype(np.complex64)
        fs_out = fs_out * ratio.numerator / ratio.denominator
        resampled = f", resampled to {TARGET_SPS} samples/symbol"
    return y, fs_out, decim, resampled


def classify_center(classifier, iq):
    """(label, confidence %) of the middle 1024-sample frame of iq, or None without a classifier."""
    if classifier is None:
        return None
    start = max(0, len(iq) // 2 - FRAME_LEN // 2)
    return classifier.predict(iq[start : start + FRAME_LEN])[0]


def compute_spectra(source, fs, nfft):
    """PSD and waterfall over the whole memory-mapped file (streamed) plus the centre window."""
    n = len(source)
    center = source.read(max(0, n // 2 - CENTER_SAMPLES // 2), CENTER_SAMPLES)
    freqs, psd = stream_psd(source, fs, nperseg=nfft)
    wf_freqs, wf_times, wf = stream_spectrogram(source, fs, nperseg=nfft, rows=WATERFALL_FRAMES)
    return {"n": n, "center": center, "freqs": freqs, "psd": psd, "wf_freqs": wf_freqs, "wf_times": wf_times,
            "wf": wf.astype(np.float32), "samples": 2 * min(n, STREAM_BUDGET) + center.size}


def compute_constellation_view(iq, fs, label):
    """CFO-corrected constellation cloud for a signal classified as `label`."""
    corrected, cfo = derotate(iq, fs, label)
    ci, cq = compute_constellation(corrected)
    return {"cfo_hz": cfo, "ci": ci, "cq": cq}


def compute_signal_info(center, fs, classifier):
    snr_db, method = estimate_snr_safe(center, fs)  # always a float: 0.0 dB for noise / clipping
    info = {"snr_db": snr_db, "snr_method": method, "label": None, "confidence": None, "classify_error": None}
    try:
        result = classify_center(classifier, center)
        if result is not None:
            info["label"], info["confidence"] = result
    except Exception as exc:
        info["classify_error"] = str(exc)
    info.update(compute_constellation_view(center, fs, info["label"]))
    return info


def compute_channels(source, fs, psd_data, threshold, classifier):
    """CFAR detection on the PSD, then a modulation label for each channel from the file's centre."""
    freqs, psd = psd_data
    found = detect_channels(freqs, psd, threshold_db=threshold, max_channels=MAX_CHANNELS)
    n = len(source)
    window = source.read(max(0, n // 2 - CHANNEL_SCAN_SAMPLES // 2), CHANNEL_SCAN_SAMPLES)
    for c in found:
        c["label"] = None
        if classifier is None:
            continue
        try:
            y, _, _, _ = band_signal(window, fs, c["f_lo"], c["f_hi"])
            if y.size >= FRAME_LEN:
                c["label"] = classify_center(classifier, y)[0]
        except (ValueError, RuntimeError):
            pass
    return {"found": found, "samples": window.size}


def compute_roi(source, fs, roi, symrate_text, classifier):
    """Time slice [t_lo, t_hi] of the file, band-limited to [f_lo, f_hi], with its classification and cloud."""
    f_lo, f_hi, t_lo, t_hi = roi["f_lo"], roi["f_hi"], roi["t_lo"], roi["t_hi"]
    start = int(max(t_lo, 0) * fs)
    raw = source.read(start, min(int((t_hi - t_lo) * fs), ROI_MAX_SAMPLES))
    if raw.size < 256:
        raise ValueError("ROI covers too little time; make it taller")
    try:
        hint = parse_rate(symrate_text) if symrate_text else None
    except ValueError:
        hint = None
    y, fs_out, decim, resampled = band_signal(raw, fs, f_lo, f_hi, hint)
    if y.size < FRAME_LEN:
        raise ValueError("ROI is too small: fewer than 1024 samples after band-limiting")
    note = (f"ROI {fmt_freq(f_lo)} to {fmt_freq(f_hi)}, {fmt_time(t_lo)} to {fmt_time(t_hi)}: "
            f"{y.size:,} samples at {fmt_freq(fs_out)}" + (f" (decimated x{decim})" if decim > 1 else "") + resampled)
    result = {"y": y, "fs": fs_out, "note": note, "label": None, "confidence": None, "classify_error": None,
              "samples": raw.size}
    try:
        found = classify_center(classifier, y)
        if found is not None:
            result["label"], result["confidence"] = found
    except Exception as exc:
        result["classify_error"] = str(exc)
    result.update(compute_constellation_view(y[:CENTER_SAMPLES], fs_out, result["label"]))
    return result


def compute_metrics(baseband, fs, scheme, symbol_rate):
    """EVM / phase noise of the sliced symbols, with the points to draw."""
    if scheme not in LINEAR_MODULATIONS:
        return {"note": f"EVM: n/a | {scheme} has no constellation"}
    try:
        symbols = recover_constellation(baseband, fs, scheme, symbol_rate)
        metrics = constellation_metrics(symbols, scheme)
    except ValueError as exc:
        return {"note": f"EVM: n/a | {exc}"}
    pts = symbols[min(len(symbols) // 5, 100):]
    pts = pts[:: -(-len(pts) // MAX_SCATTER_POINTS)]  # stride slice: a view, at most MAX_SCATTER_POINTS points
    scale = 1.0 / max(float(np.max(np.abs(pts))), 1e-9)
    return {"metrics": metrics, "pts": pts * scale, "ideal": ideal_constellation(scheme) * scale}


def compute_pipeline(iq, fs, opts, label):
    """Symbol rate -> CFO -> demodulation -> de-interleaving -> FEC, plus constellation metrics."""
    try:
        text = opts["symrate_text"]
        rate_note = None
        if text:
            symbol_rate = parse_rate(text)
        else:
            symbol_rate, score = estimate_symbol_rate(iq, fs)
            rate_note = f"Symbol rate: {symbol_rate:,.1f} baud (estimated, {score:.0f} dB)"
        scheme = opts["scheme"]
        if scheme.startswith("Auto"):
            scheme = label
            if scheme not in SUPPORTED:
                raise ValueError(f"Auto-detected '{scheme}' cannot be demodulated; pick a scheme manually.")
        baseband, cfo = derotate(iq[:DEMOD_MAX_SAMPLES], fs, scheme)
        cfo_note = f"CFO removed: {cfo:,.1f} Hz" if cfo is not None else None
        stages, notes = run_baseband(baseband, fs, scheme, symbol_rate, opts["interleaver"], opts["fec"],
                                     skip_bits=opts["skip"], invert=opts["invert"])
        metrics = compute_metrics(baseband, fs, scheme, symbol_rate)
    except Exception as exc:
        return {"error": str(exc), "samples": min(iq.size, DEMOD_MAX_SAMPLES)}
    return {"stages": stages, "notes": notes, "symbol_rate": symbol_rate, "scheme": scheme, "estimated": not text,
            "bit_notes": [f"Scheme: {scheme}"] + ([rate_note] if rate_note else []) + ([cfo_note] if cfo_note else [])
            + notes, "metrics": metrics, "samples": min(iq.size, DEMOD_MAX_SAMPLES)}


def render_frames_text(frames, marker, error, mode, post, has_bits):
    """(header text, payload text) for the two viewers."""
    if error:
        return error, ""
    if marker is None:
        return ("No sync marker found." if has_bits else "Press Run Analysis to search for sync markers."), ""
    applied = []
    if post["descramble_scheme"]:
        where = "whole stream" if post["scope"] == "stream" else "after sync"
        applied.append(f"descrambled {post['descramble_scheme']} ({where})")
    if post["invert"]:
        applied.append("bits inverted")
    head = [f"Sync marker {marker}: {len(frames)} frame(s)"]
    if applied:
        head.append("Post-processing: " + ", ".join(applied))
    body = []
    for n, f in enumerate(frames[:MAX_FRAMES_SHOWN]):
        fields = "  ".join(f"{k}={v} (0x{v:X})" for k, v in f.header.items()) or "(header truncated)"
        extra = f"  [{'; '.join(f.notes)}]" if f.notes else ""
        head.append(f"#{n}  bit {f.start}\n    {fields}{extra}")
        body.append(f"--- Frame {n}: {len(f.payload)} bytes ({mode}) ---\n{f.render(mode)}")
    if len(frames) > MAX_FRAMES_SHOWN:
        head.append(f"... {len(frames) - MAX_FRAMES_SHOWN} more not shown")
    return "\n".join(head), "\n\n".join(body)


def compute_bit_view(bits, opts):
    """Binary / hex text, sync-word correlation and frame parsing of one bit stream."""
    try:
        marker, frames = correlate_frames(bits, opts["markers"], opts["max_errors"], **opts["post"])
        error = None
    except ValueError as exc:
        marker, frames, error = None, [], str(exc)
    header, payload = render_frames_text(frames, marker, error, opts["mode"], opts["post"], True)
    return {"n_bits": len(bits), "bin_text": format_binary(bits), "hex_text": format_hex(bits), "marker": marker,
            "frames": frames, "header_text": header, "payload_text": payload, "error": error}


def load_job(report, ctx):
    path = ctx["path"]
    if ctx["is_wav"]:
        wav_iq, fs = read_wav_file(path)
        source = IQSource.from_array(wav_iq, fs)
    else:
        fs = ctx["fs"] or detect_sample_rate_from_name(path)
        if fs is None:
            raise ValueError("Enter the sample rate first (it could not be detected from the filename).")
        source = open_iq_memmap(path, dtype=ctx["dtype"], sample_rate=fs)
    if len(source) < 2:
        source.close()
        raise ValueError("No samples were read.")
    return {"source": source, "fs": fs, "path": path, "iq": source.read(0, WINDOW_SAMPLES)}


def analysis_job(report, ctx):
    """Spectra -> signal info -> channels -> (ROI) -> (baseband pipeline), publishing each step as it finishes."""
    src, fs, classifier = ctx["source"], ctx["fs"], ctx["classifier"]
    samples = 0
    spectra = compute_spectra(src, fs, ctx["nfft"])
    samples += spectra["samples"]
    report("spectra", spectra)
    info = compute_signal_info(spectra["center"], fs, classifier)
    report("signal", info)
    if ctx["scan_channels"]:
        channels = compute_channels(src, fs, (spectra["freqs"], spectra["psd"]), ctx["threshold"], classifier)
        samples += channels["samples"]
        report("channels", channels)
    iq, sig_fs, label = ctx["iq"], fs, info["label"]
    if ctx["roi"] is not None:
        try:
            roi = compute_roi(src, fs, ctx["roi"], ctx["opts"]["symrate_text"], classifier)
        except ValueError as exc:
            report("roi_error", str(exc))
            return {"samples": samples, "pipeline_ran": False, "auto": ctx["auto"], "name": ctx["name"], "roi_failed": True}
        samples += roi["samples"]
        report("roi", roi)
        iq, sig_fs, label = roi["y"], roi["fs"], roi["label"]
    opts = ctx["opts"]
    if ctx["auto"]:
        if classifier is None or label is None:
            return {"samples": samples, "auto_fail": ctx["classifier_error"] or "the classifier is unavailable",
                    "auto": True, "name": ctx["name"]}
        if label not in SUPPORTED:
            return {"samples": samples, "auto_fail": f"detected '{label}' has no demodulator; choose a scheme manually",
                    "auto": True, "name": ctx["name"]}
        report("auto_scheme", label)
        opts = {**opts, "scheme": label, "symrate_text": ""}  # blank -> the pipeline estimates the baud rate
    if ctx["pipeline"]:
        pipe = compute_pipeline(iq, sig_fs, opts, label)
        pipe["fill_rate"] = ctx["roi"] is None
        samples += pipe["samples"]
        report("pipeline", pipe)
    return {"samples": samples, "pipeline_ran": ctx["pipeline"], "auto": ctx["auto"], "label": label, "name": ctx["name"]}


def roi_job(report, ctx):
    try:
        roi = compute_roi(ctx["source"], ctx["fs"], ctx["roi"], ctx["opts"]["symrate_text"], ctx["classifier"])
    except ValueError as exc:
        report("roi_error", str(exc))
        return {"samples": 0}
    report("roi", roi)
    pipe = compute_pipeline(roi["y"], roi["fs"], ctx["opts"], roi["label"])
    pipe["fill_rate"] = False  # an ROI estimate would go stale when the ROI moves
    report("pipeline", pipe)
    return {"samples": roi["samples"] + pipe["samples"]}


def channels_job(report, ctx):
    try:
        result = compute_channels(ctx["source"], ctx["fs"], ctx["psd_data"], ctx["threshold"], ctx["classifier"])
    except ValueError as exc:
        return {"error": str(exc), "samples": 0}
    return result


def audio_job(report, ctx):
    y, fs_out, _ = stream_baseband(ctx["source"], ctx["fs"], ctx["f_lo"], ctx["f_hi"], ctx["start"], ctx["count"],
                                   oversample=8.0)
    audio, rate = demodulate_audio(y, fs_out, ctx["mode"])
    return {"pcm": to_pcm16(audio), "rate": rate, "samples": ctx["count"],
            "note": f"{ctx['mode']}, {ctx['where']}, {audio.size / rate:.1f} s"}


def pdf_job(report, path, data):
    build_report(path, data)
    return {"path": path, "samples": 0}


def live_bits_job(report, ctx):
    """Classify and demodulate the samples captured while the squelch was open."""
    iq, fs = ctx["iq"], ctx["fs"]
    label = None
    try:
        found = classify_center(ctx["classifier"], iq)
        label = found[0] if found else None
    except Exception:
        pass
    pipe = compute_pipeline(iq, fs, ctx["opts"], label)
    pipe["fill_rate"] = False  # keep the user's symbol-rate box as typed
    report("pipeline", pipe)
    return {"samples": pipe["samples"]}


def bit_view_job(report, bits, opts):
    return compute_bit_view(bits, opts)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NTRO Signal Analyzer")
        self.resize(1300, 900)
        self.file_path = None
        self.source = None  # lazy IQSource over the whole file
        self.iq = None  # head window loaded from source
        self.center = None
        self.sample_rate = None
        self.classifier = None
        self.classifier_error = None
        self.detected_label = None
        self.bit_notes = []
        self.stage_bits = {}
        self.frames = []
        self.sync_used = None
        self.symbol_rate = None
        self.scheme_used = None
        self.cfo_hz = None
        self.confidence = None
        self.snr_db = 0.0
        self.snr_method = "default"
        try:
            self.classifier = ModulationClassifier()
        except Exception as exc:
            self.classifier_error = str(exc)

        pg.setConfigOptions(antialias=False, background=BG, foreground="#aab0bf")

        self._main_task = None  # the one running load / analysis / ROI / audio / PDF task
        self._corr_task = None  # bit-stream correlation task (latest request wins)
        self._live_task = None  # live bit extraction while the squelch is open
        self.squelch = SquelchGate(SQUELCH_DEFAULT_DB)
        self._gate_start = 0
        self._last_extract = 0.0
        self._bit_summary_base = ""
        self._threads = set()
        self._roi_pending = False
        self._corr_seq = 0
        self._corr_dirty = False
        self._auto_report = None
        self._pending_chan_key = None
        self.live_active = False
        self.channels = []
        self.channel_curves = []
        self.channel_texts = []
        self._chan_key = None
        self.live_worker = None
        self.rec_saved = None
        self._live_center_hz = 0.0
        self.live_ring = None
        self.wf_rows = None
        self.live_chunks = 0
        self._live_fs = None
        self._op_samples = 0
        self.cpu_meter = CpuMeter()
        self.audio_sink = None
        self.audio_buffer = None
        self.audio_bytes = None
        self.audio_pcm = b""
        self._build_plots()
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_spectral_tab(), "Spectral Analysis")
        self.tabs.addTab(self._build_bitstream_tab(), "Bit Stream && Payload")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self._build_live_toolbar()
        self._build_help_menu()
        self._presentation_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F11"), self)
        self._presentation_shortcut.activated.connect(self.run_presentation_mode)
        self.statusBar().showMessage("Open an .iq or .wav file to begin.")

    def _build_help_menu(self):
        help_menu = self.menuBar().addMenu("&Help")
        diagnostics_action = QtGui.QAction("System Diagnostics", self)
        diagnostics_action.triggered.connect(self.show_system_diagnostics)
        help_menu.addAction(diagnostics_action)

    def show_system_diagnostics(self):
        """Help > System Diagnostics: a quick architecture summary for demos/handoffs."""
        html = """
        <h3>NTRO Signal Analyzer &mdash; System Diagnostics</h3>
        <table cellspacing="6">
        <tr><td><b>DSP Engine</b></td><td>Vectorized NumPy / SciPy Signal STFT</td></tr>
        <tr><td><b>ML Classifier</b></td><td>1D-ResNet (15 Epochs, 90%+ Accuracy)</td></tr>
        <tr><td><b>GUI Framework</b></td><td>PyQt6 / pyqtgraph (60 FPS Hardware Render)</td></tr>
        <tr><td><b>Supported Modulations</b></td><td>2-FSK, 4-FSK, BPSK, QPSK, 16-QAM, 64-QAM</td></tr>
        <tr><td><b>Protocol FEC</b></td><td>PN15 Multiplicative Descrambler &amp; Frame Sync (0xEB90)</td></tr>
        </table>
        """
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("System Diagnostics")
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(html)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.exec()

    def run_presentation_mode(self):
        """F11: load the bundled demo capture, run Auto-Analyze, and jump to the payload view."""
        demo_path = ROOT / "demo_samples" / "01_clean_qpsk_telemetry.iq"
        if not demo_path.is_file():
            self.statusBar().showMessage(f"Presentation mode: demo file not found at {demo_path}")
            return
        self.load_file(demo_path)
        self.wait_for_tasks()
        self.auto_analyze()
        self.wait_for_tasks()
        self.tabs.setCurrentIndex(1)

    @staticmethod
    def _stack(items, min_height):
        """Scrollable full-width vertical stack of titled widgets; each expands with the window
        but never shrinks below min_height, so views cannot squeeze each other."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(18, 14, 18, 18)
        v.setSpacing(6)
        for title, widget, height in items:
            label = QtWidgets.QLabel(title.upper())
            label.setProperty("muted", True)
            v.addWidget(label)
            widget.setMinimumHeight(height or min_height)
            v.addWidget(widget, 1)
            v.addSpacing(12)
        scroll.setWidget(body)
        return scroll

    def _constellation_block(self):
        block = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(block)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(self.metrics_label)
        v.addWidget(self.const_plot, 1)
        return block

    # ---- constellation quality (EVM / phase noise) ------------------------------------
    def _clear_metrics(self, note="EVM: -- | SNR: -- | Phase Noise: -- (run the analysis on a PSK/QAM signal)"):
        self.constellation_metrics = None
        self.ideal_scatter.setData([], [])
        self.metrics_label.setText(f'<span style="color:{MUTED}">{note}</span>')

    def _build_spectral_tab(self):
        return self._stack(
            [
                ("Spectrum (PSD)", self.psd_plot, 190),
                ("Waterfall (spectrogram) - drag the green box to select a region", self.wf_layout, 230),
                ("Constellation", self._constellation_block(), 340),
            ],
            190,
        )

    def _build_post_panel(self):
        panel = QtWidgets.QGroupBox("PAYLOAD POST-PROCESSING")
        g = QtWidgets.QGridLayout(panel)
        g.setContentsMargins(12, 18, 12, 12)
        g.setHorizontalSpacing(12)
        self.descramble_check = QtWidgets.QCheckBox("Descramble")
        self.descramble_check.setToolTip("Undo an LFSR scrambler on the frame bits (header and payload)")
        self.descramble_combo = QtWidgets.QComboBox()
        self.descramble_combo.addItems(list(DESCRAMBLE_SCHEMES))  # PN15 first
        self.descramble_combo.setToolTip("PN15 x15+x14+1, PN23 x23+x18+1, CCSDS x8+x7+x5+x3+1 (additive); "
                                         "G3RUH x17+x12+1 (self-synchronizing)")
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItems(["After sync marker", "Whole stream"])
        self.scope_combo.setToolTip("After sync marker: the sync word is sent unscrambled and the LFSR restarts "
                                    "each frame. Whole stream: descramble everything before searching for sync.")
        self.post_invert_check = QtWidgets.QCheckBox("Invert bits")
        self.post_invert_check.setToolTip("Flip the frame bits (applied before descrambling)")
        self.render_combo = QtWidgets.QComboBox()
        self.render_combo.addItems(list(PAYLOAD_MODES))
        self.render_combo.setToolTip("How payload bytes are shown: printable ASCII, UTF-8 text, or a raw hex dump")
        g.addWidget(self.descramble_check, 0, 0)
        g.addWidget(self.descramble_combo, 0, 1)
        g.addWidget(self.scope_combo, 0, 2)
        g.addWidget(self.post_invert_check, 1, 0)
        g.addWidget(self._muted("Payload view"), 1, 1)
        g.addWidget(self.render_combo, 1, 2)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 1)
        for signal in (
            self.descramble_check.toggled,
            self.descramble_combo.currentTextChanged,
            self.scope_combo.currentTextChanged,
            self.post_invert_check.toggled,
            self.render_combo.currentTextChanged,
        ):
            signal.connect(self._refresh_bit_view)
        return panel

    def _post_options(self):
        return {
            "descramble_scheme": self.descramble_combo.currentText() if self.descramble_check.isChecked() else None,
            "invert": self.post_invert_check.isChecked(),
            "scope": "stream" if self.scope_combo.currentIndex() == 1 else "frame",
        }

    def _build_bitstream_tab(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(18, 14, 18, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Stage"))
        self.stage_combo = QtWidgets.QComboBox()
        self.stage_combo.setMinimumWidth(200)
        self.stage_combo.currentTextChanged.connect(self._refresh_bit_view)
        top.addWidget(self.stage_combo)
        self.bit_summary = QtWidgets.QLabel("Press Run Analysis to extract bits.")
        self.bit_summary.setProperty("muted", True)
        self.bit_summary.setWordWrap(True)
        top.addWidget(self.bit_summary, 1)
        v.addLayout(top)

        v.addWidget(self._build_post_panel())

        items = []
        for title, attr in (
            ("Demodulated Binary", "bin_view"),
            ("Hex View", "hex_view"),
            ("Header Fields", "header_view"),
            ("Extracted Payload (ASCII / Hex)", "payload_view"),
        ):
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            setattr(self, attr, edit)
            items.append((title, edit, 150))
        v.addWidget(self._stack(items, 150), 1)
        return page

    def _build_sidebar(self):
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(350)
        outer = QtWidgets.QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        v = QtWidgets.QVBoxLayout(inner)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        title = QtWidgets.QLabel("Signal Analyzer")
        title.setObjectName("title")
        v.addWidget(title)
        v.addWidget(self._build_status_panel())
        v.addWidget(self._build_file_panel())
        v.addWidget(self._build_preset_panel())
        v.addWidget(self._build_baseband_panel())

        self.auto_btn = QtWidgets.QPushButton("One-Click Auto-Analyze")
        self.auto_btn.setObjectName("success")
        self.auto_btn.setEnabled(False)
        self.auto_btn.setToolTip("Detect modulation, estimate baud rate, then demodulate and search for sync markers")
        self.auto_btn.clicked.connect(self.auto_analyze)
        v.addWidget(self.auto_btn)

        self.run_btn = QtWidgets.QPushButton("Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_analysis)
        v.addWidget(self.run_btn)

        self.export_btn = QtWidgets.QPushButton("Export Results (JSON + CSV)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        v.addWidget(self.export_btn)

        self.pdf_btn = QtWidgets.QPushButton("Export PDF Report")
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.setToolTip("Metadata, PSD and constellation snapshots, AMC result and the decoded payload")
        self.pdf_btn.clicked.connect(self.export_pdf)
        v.addWidget(self.pdf_btn)

        v.addWidget(self._build_info_panel())
        v.addWidget(self._build_channel_panel())
        v.addWidget(self._build_roi_panel())
        v.addWidget(self._build_audio_panel())
        v.addWidget(self._build_perf_panel())

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setProperty("muted", True)
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)
        v.addStretch(1)
        for widget in side.findChildren((QtWidgets.QSpinBox, QtWidgets.QPushButton)):
            widget.setMinimumWidth(70)
        for combo in side.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
        return side

    def _build_file_panel(self):
        panel = QtWidgets.QGroupBox("FILE SELECTION")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(4)
        self.open_btn = QtWidgets.QPushButton("Open File (.iq / .wav)")
        self.open_btn.clicked.connect(self.pick_file)
        v.addWidget(self.open_btn)
        self.file_label = QtWidgets.QLabel("No file selected")
        self.file_label.setProperty("muted", True)
        self.file_label.setWordWrap(True)
        v.addWidget(self.file_label)

        v.addWidget(self._muted("IQ sample format"))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["int16", "float32"])
        v.addWidget(self.format_combo)

        v.addWidget(self._muted("Sample rate (Hz, e.g. 2.4M)"))
        self.rate_edit = QtWidgets.QLineEdit()
        self.rate_edit.setPlaceholderText("2.4M")
        self.rate_edit.returnPressed.connect(self.run_analysis)
        v.addWidget(self.rate_edit)

        v.addWidget(self._muted("FFT size"))
        self.fft_combo = QtWidgets.QComboBox()
        self.fft_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.fft_combo.setCurrentText("1024")
        v.addWidget(self.fft_combo)
        return panel

    def _build_preset_panel(self):
        panel = QtWidgets.QGroupBox("PROTOCOL PRESETS")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(4)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        v.addWidget(self.preset_combo)
        self.preset_note = self._muted("")
        v.addWidget(self.preset_note)
        return panel

    def _build_status_panel(self):
        panel = QtWidgets.QGroupBox("STATUS")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)

        def led_row(name):
            row = QtWidgets.QHBoxLayout()
            led = Led()
            row.addWidget(led)
            row.addWidget(self._muted(name), 1)
            value = QtWidgets.QLabel("")
            value.setObjectName("ledText")
            row.addWidget(value)
            v.addLayout(row)
            return led, value

        self.lock_led, self.lock_text = led_row("Signal Lock")
        self.lock_led.setToolTip("Green when a sync marker is found in the extracted bit stream")
        self.engine_led, self.engine_text = led_row("DSP Engine")
        self.squelch_led, self.squelch_text = led_row("Squelch")
        self.squelch_led.setToolTip("Live stream only: green while a signal is above the squelch threshold")

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._muted("SNR quality"), 1)
        self.snr_quality = QtWidgets.QLabel("No estimate")
        self.snr_quality.setObjectName("ledText")
        row.addWidget(self.snr_quality)
        v.addLayout(row)
        self.snr_bar = QualityBar()
        self.snr_bar.setToolTip("0 to 30 dB: red < 6, amber < 12, cyan < 20, green above")
        v.addWidget(self.snr_bar)

        self.set_lock(False)
        self.set_engine_state(False)
        self._set_squelch_tag(None)
        return panel

    def _set_squelch_tag(self, is_open, level_db=None):
        """None = no live stream, False = closed (bit extraction paused), True = open."""
        if is_open is None:
            color, text = "#475569", "--"
        elif is_open:
            color, text = GREEN, "OPEN / LOCK"
        else:
            color, text = "#475569", "CLOSED"
        self.squelch_led.set_color(color)
        self.squelch_text.setText(text)
        self.squelch_text.setStyleSheet(f"color: {GREEN if is_open else MUTED};")
        self.squelch_text.setToolTip("" if level_db is None else f"Peak level {level_db:.1f} dB, "
                                     f"threshold {self.squelch.threshold_db:.0f} dB")

    def set_lock(self, locked, marker=None):
        self.lock_led.set_color(GREEN if locked else RED)
        self.lock_text.setText(f"SYNC {marker}" if locked else "NO SYNC")
        self.lock_text.setStyleSheet(f"color: {GREEN if locked else RED};")

    def _build_baseband_panel(self):
        panel = QtWidgets.QGroupBox("BASEBAND PROCESSING")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(4)

        v.addWidget(self._muted("Demodulation scheme"))
        self.demod_combo = QtWidgets.QComboBox()
        self.demod_combo.addItems(["Auto (from classifier)", *SUPPORTED])
        v.addWidget(self.demod_combo)

        v.addWidget(self._muted("Symbol rate (baud, e.g. 250k)"))
        self.symrate_edit = QtWidgets.QLineEdit()
        self.symrate_edit.setPlaceholderText("blank = auto-estimate")
        v.addWidget(self.symrate_edit)

        row = QtWidgets.QHBoxLayout()
        self.skip_spin = QtWidgets.QSpinBox()
        self.skip_spin.setRange(0, 100000)
        self.skip_spin.setPrefix("skip ")
        self.skip_spin.setToolTip("Bits to drop from the start (frame alignment)")
        self.invert_check = QtWidgets.QCheckBox("Invert")
        self.invert_check.setToolTip("Flip all bits (resolves BPSK phase ambiguity)")
        row.addWidget(self.skip_spin, 1)
        row.addWidget(self.invert_check)
        v.addLayout(row)

        v.addSpacing(4)
        v.addWidget(self._muted("De-interleaver mode"))
        self.interleave_combo = QtWidgets.QComboBox()
        self.interleave_combo.addItems(list(INTERLEAVERS))
        self.interleave_combo.currentTextChanged.connect(self._interleaver_changed)
        v.addWidget(self.interleave_combo)

        params = QtWidgets.QHBoxLayout()
        self.il_a_label, self.il_b_label = QtWidgets.QLabel("A"), QtWidgets.QLabel("B")
        self.il_a_spin, self.il_b_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for label, spin in ((self.il_a_label, self.il_a_spin), (self.il_b_label, self.il_b_spin)):
            spin.setRange(1, 100000)
            label.setProperty("muted", True)
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(label)
            col.addWidget(spin)
            params.addLayout(col)
        v.addLayout(params)
        self._interleaver_changed(self.interleave_combo.currentText())

        v.addSpacing(4)
        v.addWidget(self._muted("FEC decoder"))
        self.fec_combo = QtWidgets.QComboBox()
        self.fec_combo.addItems(list(FEC_DECODERS))
        v.addWidget(self.fec_combo)

        v.addSpacing(4)
        v.addWidget(self._muted("SYNC WORD CORRELATION"))
        v.addWidget(self._muted("Hex sync markers (comma separated)"))
        self.sync_edit = QtWidgets.QLineEdit(DEFAULT_SYNC)
        self.sync_edit.setToolTip("e.g. 0x1ACFFC1D, 0xEB90, 0x7EA5. The marker with the most hits is used.")
        self.sync_edit.editingFinished.connect(self._refresh_bit_view)
        v.addWidget(self.sync_edit)
        self.sync_err_spin = QtWidgets.QSpinBox()
        self.sync_err_spin.setRange(0, 8)
        self.sync_err_spin.setPrefix("max errors ")
        self.sync_err_spin.setToolTip("Hamming distance tolerated when matching the marker")
        self.sync_err_spin.valueChanged.connect(self._refresh_bit_view)
        v.addWidget(self.sync_err_spin)

        self._applying_preset = False
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        # editing a preset-controlled field by hand switches back to Custom
        for signal in (
            self.demod_combo.currentTextChanged,
            self.symrate_edit.textEdited,
            self.interleave_combo.currentTextChanged,
            self.il_a_spin.valueChanged,
            self.il_b_spin.valueChanged,
            self.sync_edit.textEdited,
        ):
            signal.connect(self._preset_edited)
        return panel

    def apply_preset(self, name):
        preset = PRESETS.get(name)
        self.preset_note.setText(preset["note"] if preset else "")
        if preset is None:
            return
        self._applying_preset = True
        try:
            self.demod_combo.setCurrentText(preset["demod"])
            self.symrate_edit.setText(preset["baud"])
            self.interleave_combo.setCurrentText(preset["interleaver"])  # resets the A/B spins
            if preset["il"] is not None:
                self.il_a_spin.setValue(preset["il"][0])
                self.il_b_spin.setValue(preset["il"][1])
            self.sync_edit.setText(preset["sync"])
        finally:
            self._applying_preset = False
        self._refresh_bit_view()

    def _preset_edited(self, *_):
        if not self._applying_preset and self.preset_combo.currentText() != "Custom":
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
            self.preset_note.setText("")

    def _interleaver_changed(self, name):
        spec = INTERLEAVERS[name]
        enabled = spec is not None
        for w in (self.il_a_label, self.il_b_label, self.il_a_spin, self.il_b_spin):
            w.setEnabled(enabled)
        if enabled:
            self.il_a_label.setText(spec[1])
            self.il_b_label.setText(spec[2])
            self.il_a_spin.setValue(spec[3])
            self.il_b_spin.setValue(spec[4])
        else:
            self.il_a_label.setText("A")
            self.il_b_label.setText("B")

    def _build_info_panel(self):
        panel = QtWidgets.QGroupBox("SIGNAL INFORMATION")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(2)

        def metric(name):
            v.addSpacing(4)
            v.addWidget(self._muted(name))
            value = QtWidgets.QLabel("--")
            value.setObjectName("metricValue")
            v.addWidget(value)
            return value

        self.mod_value = metric("Detected Modulation (AMC)")
        self.conf_value = metric("Confidence Score")
        self.rate_value = metric("Estimated Sample Rate")
        self.snr_value = metric("Calculated SNR")
        self.cfo_value = metric("Carrier Offset (CFO)")
        return panel

    @staticmethod
    def _muted(text):
        label = QtWidgets.QLabel(text)
        label.setProperty("muted", True)
        label.setWordWrap(True)
        return label

    def _build_plots(self):
        self.psd_plot = pg.PlotWidget()
        style_plot(self.psd_plot.getPlotItem(), "Power Spectral Density", "Frequency (Hz)", "Power (dB/Hz)")
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen(TRACE, width=1.2), autoDownsample=True, clipToView=True)

        self.wf_layout = pg.GraphicsLayoutWidget()
        self.wf_plot = self.wf_layout.addPlot(row=0, col=0)
        style_plot(self.wf_plot, "Waterfall", "Frequency (Hz)", "Time (s)")
        self.wf_image = pg.ImageItem(autoDownsample=True)
        self.wf_plot.addItem(self.wf_image)
        self.wf_plot.setMouseEnabled(x=True, y=True)
        self.wf_hist = pg.HistogramLUTItem(image=self.wf_image)
        self.wf_hist.gradient.loadPreset("viridis")
        self.wf_layout.addItem(self.wf_hist, row=0, col=1)

        self.chan_region = pg.LinearRegionItem(values=(0, 1), movable=False,
                                               brush=pg.mkBrush(34, 211, 238, 45), pen=pg.mkPen(ACCENT))
        self.chan_region.setZValue(5)
        self.chan_region.hide()
        self.psd_plot.addItem(self.chan_region, ignoreBounds=True)
        self.psd_data = None  # (freqs, psd_db) for the hover read-out
        self.wf_data = None  # (matrix, f0, df, t0, dt)
        self.psd_cross = Crosshair(self.psd_plot.getPlotItem(), self.psd_plot, self._describe_psd)
        self.wf_cross = Crosshair(self.wf_plot, self.wf_layout, self._describe_wf)
        for plot_item in (self.psd_plot.getPlotItem(), self.wf_plot):  # double-click = retune (live only)
            plot_item.scene().sigMouseClicked.connect(lambda ev, p=plot_item: self._on_plot_clicked(ev, p))

        self._roi_file = None
        self.roi_mode = False
        self.roi_iq = None
        self.roi_fs = None
        self._roi_pending = False
        self.roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen(GREEN, width=2), handlePen=pg.mkPen(GREEN),
                              handleHoverPen=pg.mkPen(ACCENT), rotatable=False)
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.setZValue(900)
        self.roi.hide()
        self.wf_plot.addItem(self.roi)
        self.roi_timer = QtCore.QTimer(self)
        self.roi_timer.setSingleShot(True)
        self.roi_timer.setInterval(120)
        self.roi_timer.timeout.connect(self._apply_roi)
        self.roi.sigRegionChanged.connect(self._roi_moved)

        self.const_plot = pg.PlotWidget()
        style_plot(self.const_plot.getPlotItem(), "IQ Constellation", "I", "Q")
        self.const_plot.setAspectLocked(True)
        self.const_plot.setRange(xRange=(-1.1, 1.1), yRange=(-1.1, 1.1))
        for view in (self.psd_plot.getPlotItem(), self.wf_plot, self.const_plot.getPlotItem()):
            view.setClipToView(True)  # only the visible part of a curve is drawn
            view.setDownsampling(auto=True, mode="peak")  # one min/max pair per screen pixel
        self.wf_image.setAutoDownsample(True)  # the image is sampled down to the screen resolution
        self.const_scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(34, 211, 238, 110))
        self.const_plot.addItem(self.const_scatter)
        self.ideal_scatter = pg.ScatterPlotItem(size=11, symbol="+", pen=pg.mkPen("#f5f7fa", width=1.5), brush=None)
        self.ideal_scatter.setZValue(10)
        self.const_plot.addItem(self.ideal_scatter)
        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setObjectName("metricsBar")
        self.metrics_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setToolTip(
            "EVM: RMS error vector against the nearest ideal point, relative to RMS signal power.\n"
            "SNR: implied by the EVM (all impairments counted as noise).\n"
            "Phase Noise: standard deviation of the symbol phase error; Offset: its mean (residual rotation)."
        )
        self.constellation_metrics = None
        self._clear_metrics()


    # ---- live SDR input --------------------------------------------------------
    def _build_live_toolbar(self):
        bar = QtWidgets.QToolBar("Live SDR")
        bar.setObjectName("liveBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        bar.addWidget(QtWidgets.QLabel("LIVE SDR"))
        self.live_backend = QtWidgets.QComboBox()
        self.live_backend.addItems(list(LIVE_BACKENDS))
        self.live_backend.setMinimumWidth(170)
        for i, name in enumerate(LIVE_BACKENDS):
            ok, why = backend_available(name)
            self.live_backend.setItemData(i, "available" if ok else why, QtCore.Qt.ItemDataRole.ToolTipRole)
        first_real = next((b for b in LIVE_BACKENDS[:2] if backend_available(b)[0]), LIVE_BACKENDS[2])
        self.live_backend.setCurrentText(first_real)
        bar.addWidget(self.live_backend)

        bar.addWidget(QtWidgets.QLabel("Center Frequency"))
        self.live_freq = QtWidgets.QDoubleSpinBox()
        self.live_freq.setRange(0.001, 6000.0)
        self.live_freq.setDecimals(4)
        self.live_freq.setValue(100.0)
        self.live_freq.setSuffix(" MHz")
        self.live_freq.setKeyboardTracking(False)
        bar.addWidget(self.live_freq)

        bar.addWidget(QtWidgets.QLabel("Gain"))
        self.live_gain = QtWidgets.QDoubleSpinBox()
        self.live_gain.setRange(-1.0, 70.0)
        self.live_gain.setSingleStep(1.0)
        self.live_gain.setValue(30.0)
        self.live_gain.setDecimals(1)
        self.live_gain.setSuffix(" dB")
        self.live_gain.setSpecialValueText("Auto")  # shown at the minimum (-1)
        self.live_gain.setKeyboardTracking(False)
        bar.addWidget(self.live_gain)

        bar.addWidget(QtWidgets.QLabel("Squelch"))
        self.squelch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.squelch_slider.setRange(-120, 0)
        self.squelch_slider.setValue(SQUELCH_DEFAULT_DB)
        self.squelch_slider.setFixedWidth(130)
        self.squelch_slider.setToolTip(
            "Bit-stream extraction runs only while the strongest spectral line of the live signal is above this "
            "level (dB relative to full scale, the same scale as the waterfall). Set it a few dB above the noise "
            "peak shown in the status line; -120 dB keeps the squelch always open.")
        bar.addWidget(self.squelch_slider)
        self.squelch_label = QtWidgets.QLabel(f"{SQUELCH_DEFAULT_DB} dB")
        self.squelch_label.setMinimumWidth(52)
        bar.addWidget(self.squelch_label)
        self.squelch_slider.valueChanged.connect(self._squelch_slider_changed)

        bar.addWidget(QtWidgets.QLabel("Sample Rate"))
        self.live_rate = QtWidgets.QLineEdit("2.4M")
        self.live_rate.setFixedWidth(90)
        self.live_rate.setToolTip("Samples per second, e.g. 2.4M or 250k (fixed while streaming)")
        bar.addWidget(self.live_rate)

        self.live_btn = QtWidgets.QPushButton("Start Live Stream")
        self.live_btn.setObjectName("live")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self.toggle_live)
        bar.addWidget(self.live_btn)

        self.rec_btn = QtWidgets.QPushButton("Record Stream")
        self.rec_btn.setObjectName("rec")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setToolTip("Append every live IQ buffer to a .iq file in the recordings/ folder "
                                "(float32 I/Q). Toggle it before starting the stream to arm it.")
        self.rec_btn.toggled.connect(self.toggle_record)
        bar.addWidget(self.rec_btn)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        self.rec_indicator = QtWidgets.QLabel("")
        self.rec_indicator.setObjectName("recIndicator")
        self.rec_action = bar.addWidget(self.rec_indicator)
        self.rec_action.setVisible(False)
        self.rec_timer = QtCore.QTimer(self)
        self.rec_timer.setInterval(250)
        self.rec_timer.timeout.connect(self._tick_rec)
        self._rec_phase = 0
        self.live_status = QtWidgets.QLabel("Live stream idle")
        bar.addWidget(self.live_status)

        self.live_freq.valueChanged.connect(self._live_params_changed)
        self.live_gain.valueChanged.connect(self._live_params_changed)

    def toggle_live(self, checked):
        if checked:
            self.start_live()
        else:
            self.stop_live()

    def _squelch_slider_changed(self, value):
        self.squelch_label.setText(f"{value} dB")
        self.squelch.threshold_db = float(value)

    def _squelch_transition(self, is_open):
        """Gate opened or closed: mark where the burst starts, refresh the tag and the bit view note."""
        ring = self.live_ring
        if is_open:
            self._gate_start = ring.total_written
            self._last_extract = 0.0
        else:
            self._live_extract(force=True)  # one last pass over the finished burst
        self._update_bit_summary_note()

    def _live_extract(self, force=False):
        """Demodulate the newest samples of the current gated burst in the background (latest wins)."""
        ring, worker = self.live_ring, self.live_worker
        if ring is None or worker is None or self._live_task is not None:
            return
        since = ring.total_written - self._gate_start
        if since < LIVE_BITS_MIN_SAMPLES:
            return
        now = time.perf_counter()
        if not force and now - self._last_extract < LIVE_BITS_INTERVAL_S:
            return
        self._last_extract = now
        ctx = {"iq": ring.snapshot(min(since, LIVE_BITS_MAX_SAMPLES)), "fs": self._live_fs,
               "classifier": self.classifier, "opts": self._pipeline_opts()}
        self._start_task("live_bits", live_bits_job, (ctx,), {"pipeline": self._apply_live_pipeline}, slot="live")

    def _apply_live_pipeline(self, res):
        if self.live_active:  # a result that arrives after the stream stopped is stale
            self._apply_pipeline(res)

    def _update_bit_summary_note(self):
        if self.live_active and not self.squelch.is_open and self._bit_summary_base:
            self.bit_summary.setText(self._bit_summary_base + "  |  squelch closed: extraction paused")
        elif self._bit_summary_base:
            self.bit_summary.setText(self._bit_summary_base)

    def toggle_record(self, checked):
        """REC on: record from now on if streaming, otherwise arm it for the next stream. REC off: close the file."""
        self.rec_btn.setText("Recording..." if checked and self.live_active else "Record Stream (armed)" if checked
                             else "Record Stream")
        if self.live_active and self.live_worker is not None:
            self.live_worker.request(record=bool(checked))
        elif checked:
            self.statusBar().showMessage("REC armed: recording starts with the next live stream.")

    def _on_record_started(self, path):
        self._rec_phase = 0
        self.rec_action.setVisible(True)
        self.rec_timer.start()
        self._tick_rec()
        self.rec_btn.setText("Recording...")
        self.statusBar().showMessage(f"Recording to {Path(path).relative_to(ROOT) if ROOT in Path(path).parents else path}")

    def _on_record_stopped(self, path, n_bytes):
        self.rec_timer.stop()
        self.rec_action.setVisible(False)
        if not self.rec_btn.isChecked():
            self.rec_btn.setText("Record Stream")
        self.statusBar().showMessage(f"Saved {Path(path).name}: {format_bytes(n_bytes)} in {Path(path).parent}")

    def _on_record_failed(self, message):
        self.rec_timer.stop()
        self.rec_action.setVisible(False)
        self.rec_btn.blockSignals(True)
        self.rec_btn.setChecked(False)
        self.rec_btn.blockSignals(False)
        self.rec_btn.setText("Record Stream")
        self._error("Recording", ValueError(message))

    def _tick_rec(self):
        """Blink the red dot every 0.5 s and show the elapsed time and file size."""
        worker = self.live_worker
        n_bytes = worker.rec_bytes if worker is not None else 0
        fs = self._live_fs or 1.0
        seconds = int(n_bytes / 8 / fs)  # complex64 = 8 bytes per sample
        on = (self._rec_phase // 2) % 2 == 0
        self._rec_phase += 1
        color = "#ff3b3b" if on else "#5a2226"
        self.rec_indicator.setText(
            f'<span style="color:{color}">\u25cf REC</span>'
            f'<span style="color:{MUTED}; font-weight:600">  {seconds // 60}:{seconds % 60:02d}  '
            f'{format_bytes(n_bytes)}</span>'
        )

    def _set_live_button(self, on):
        self.live_btn.blockSignals(True)
        self.live_btn.setChecked(on)
        self.live_btn.blockSignals(False)
        self.live_btn.setText("Stop Live Stream" if on else "Start Live Stream")

    def start_live(self):
        if self.busy:
            self._set_live_button(False)
            self.statusBar().showMessage("Busy: wait for the current task to finish before streaming.")
            return
        self.stop_audio()
        backend = self.live_backend.currentText()
        ok, why = backend_available(backend)
        if not ok:
            self._set_live_button(False)
            self._error("Live SDR unavailable", ValueError(why + ". Choose 'Simulated' to try the live view."))
            return
        try:
            fs = parse_rate(self.live_rate.text())
        except ValueError:
            self._set_live_button(False)
            self._error("Live SDR", ValueError(f"Cannot parse sample rate {self.live_rate.text()!r}."))
            return
        center = self.live_freq.value() * 1e6
        chunk = int(np.clip(fs / 20, 16384, 262144)) // 1024 * 1024
        self.live_ring = RingBuffer(int(min(LIVE_CAPTURE_SECONDS * fs, LIVE_CAPTURE_MAX)))
        self.live_worker = LiveWorker(backend, center, fs, self.live_gain.value(), chunk, self.live_ring,
                                      record=self.rec_btn.isChecked(), record_dir=RECORDINGS_DIR)
        self.squelch.reset()
        self._gate_start = 0
        self._set_squelch_tag(False)
        self._bit_summary_base = ""
        self.stage_bits = {}
        self.bin_view.setPlainText("")
        self.hex_view.setPlainText("")
        self._show_frames([], None)
        self.bit_summary.setText("Live stream: bit extraction starts when a signal rises above the squelch threshold.")
        self.live_worker.record_started.connect(self._on_record_started)
        self.live_worker.record_stopped.connect(self._on_record_stopped)
        self.live_worker.record_failed.connect(self._on_record_failed)
        self.live_worker.chunk.connect(self._on_live_chunk)
        self.live_worker.opened.connect(self._on_live_opened)
        self.live_worker.failed.connect(self._on_live_failed)
        self._prepare_live_views(fs, center)
        self.live_active = True
        self._set_live_button(True)
        self._live_ui(True)
        self.live_status.setText(f"Connecting to {backend}...")
        self.live_worker.start()

    def _prepare_live_views(self, fs, center):
        nfft = int(self.fft_combo.currentText())
        self._live_fs, self.live_nfft, self.live_chunks = fs, nfft, 0
        self._live_center_hz = center
        self.wf_rows = np.full((nfft, LIVE_ROWS), np.nan, dtype=np.float32)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
        self.psd_curve.setData([], [])
        self.psd_plot.setXRange(freqs[0], freqs[-1], padding=0)
        self.wf_plot.setXRange(freqs[0], freqs[-1], padding=0)
        self.roi.hide()
        self._clear_channels()
        self.channel_note.setText("Channels are detected once the stream is stopped.")
        self.roi_mode, self.roi_iq = False, None
        self.roi_check.blockSignals(True)
        self.roi_check.setChecked(False)
        self.roi_check.blockSignals(False)
        self.roi_label.setText("ROI is available once the stream is stopped.")
        self._set_live_titles(center)
        self._clear_metrics()

    def _set_live_titles(self, center_hz):
        tag = f"LIVE @ {center_hz / 1e6:.4f} MHz (frequencies relative to centre)"
        self.psd_plot.getPlotItem().setTitle(f"Power Spectral Density - {tag}", color="#c9cdd8", size="11pt")
        self.wf_plot.setTitle(f"Waterfall - {tag}", color="#c9cdd8", size="11pt")

    def _reset_titles(self):
        self.psd_plot.getPlotItem().setTitle("Power Spectral Density", color="#c9cdd8", size="11pt")
        self.wf_plot.setTitle("Waterfall", color="#c9cdd8", size="11pt")

    def _on_plot_clicked(self, ev, plot_item):
        """Double-click on the PSD or waterfall: retune the receiver to the clicked frequency."""
        if not ev.double() or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = plot_item.vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        ev.accept()
        self.retune_to_offset(float(vb.mapSceneToView(ev.scenePos()).x()))

    def retune_to_offset(self, offset_hz):
        """Centre the receiver on the frequency `offset_hz` away from the current centre.

        Sets the toolbar's Center Frequency field, which sends the new centre to the running receiver.
        The plots show frequencies relative to the centre, so this only makes sense while streaming.
        """
        if not self.live_active:
            self.statusBar().showMessage("Double-click retunes the live receiver: start a live stream first.")
            return None
        lo, hi = self.live_freq.minimum(), self.live_freq.maximum()
        new_mhz = round(min(max(self.live_freq.value() + offset_hz / 1e6, lo), hi), self.live_freq.decimals())
        self.live_freq.setValue(new_mhz)  # valueChanged -> _live_params_changed -> worker.request(center_hz)
        self.statusBar().showMessage(f"Retuned to {new_mhz:.4f} MHz ({fmt_freq(offset_hz)} from the previous centre)")
        return new_mhz

    def _live_params_changed(self, *_):
        if not self.live_active or self.live_worker is None:
            return
        center = self.live_freq.value() * 1e6
        self.live_worker.request(center_hz=center, gain_db=self.live_gain.value())
        if abs(center - self._live_center_hz) > 0.5 and self.wf_rows is not None:
            self.wf_rows[:] = np.nan  # retuned: the old rows are relative to the old centre
            self.live_chunks = 0
            self._live_center_hz = center
        self._set_live_titles(center)

    def _on_live_opened(self, fs, center):
        self._live_fs = fs
        if abs(fs - parse_rate(self.live_rate.text())) > 1:
            self.live_rate.setText(f"{fs:g}")
        self.live_status.setText(f"Streaming {fs / 1e6:.4g} MS/s")

    def _finish_recording(self, worker):
        """After the worker has stopped its file is closed: hide the indicator and clear REC."""
        self.rec_timer.stop()
        self.rec_action.setVisible(False)
        if self.rec_btn.isChecked():
            self.rec_btn.blockSignals(True)
            self.rec_btn.setChecked(False)
            self.rec_btn.blockSignals(False)
        self.rec_btn.setText("Record Stream")
        if worker is not None and worker.recordings:
            path, n_bytes = worker.recordings[-1]
            self.rec_saved = path
            self.statusBar().showMessage(f"Recording saved: {Path(path).name} ({format_bytes(n_bytes)}) in {Path(path).parent}")

    def _on_live_failed(self, message):
        was_active = self.live_active
        self._set_squelch_tag(None)
        self._finish_recording(self.live_worker)
        self.live_active = False
        self._set_live_button(False)
        self._live_ui(False)
        self.live_status.setText("Live stream stopped")
        if was_active:
            self._error("Live SDR", ValueError(message))

    def _on_live_chunk(self, chunk):
        worker = self.live_worker
        if not self.live_active or worker is None:
            return
        t0 = time.perf_counter()
        fs, nfft = self._live_fs, self.live_nfft
        try:
            freqs, psd = compute_psd(chunk, fs, nperseg=nfft)
            _, row = compute_stft_row(chunk, fs, nfft)
        except ValueError:
            worker.gui_ready = True
            return
        self.psd_curve.setData(freqs, psd)
        self.psd_data = (freqs, psd)
        self.wf_rows = np.roll(self.wf_rows, -1, axis=1)
        self.wf_rows[:, -1] = row
        self.live_chunks += 1

        dt = chunk.size / fs
        df = fs / nfft
        valid = self.wf_rows[:, -min(self.live_chunks, LIVE_ROWS) :]
        floor = float(np.min(valid))
        self.wf_image.setImage(np.where(np.isfinite(self.wf_rows), self.wf_rows, floor), autoLevels=False)
        f0 = float(freqs[0] - df / 2)
        y0 = (self.live_chunks - LIVE_ROWS) * dt
        self.wf_image.setRect(QtCore.QRectF(f0, y0, df * nfft, dt * LIVE_ROWS))
        self.wf_data = (self.wf_rows, f0, float(df), float(y0), float(dt))
        self.wf_plot.setYRange(max(y0, 0.0), y0 + dt * LIVE_ROWS, padding=0)
        if self.live_chunks <= 3 or self.live_chunks % 10 == 0:
            lo, hi = np.percentile(valid, [5, 99.5])
            self.wf_hist.setLevels(float(lo), float(hi))
            self.psd_plot.setYRange(float(np.min(psd)) - 3, float(np.max(psd)) + 6, padding=0)
        level = float(np.max(row)) + DBM_OFFSET_DB  # strongest spectral line, dB rel. full scale
        is_open, changed = self.squelch.update(level, self.live_ring.total_written / fs)
        self._set_squelch_tag(is_open, level)
        if changed:
            self._squelch_transition(is_open)
        if is_open:
            self._live_extract()
        self.live_status.setText(
            f"Streaming {fs / 1e6:.4g} MS/s | peak {level:.1f} dB | {self.live_chunks} buffers | {worker.dropped} dropped"
        )
        self._op_samples = chunk.size
        self._record_perf("live buffer", time.perf_counter() - t0)
        worker.gui_ready = True

    def stop_live(self):
        if not self.live_active:
            return
        worker = self.live_worker
        self.live_active = False
        worker.stop()
        worker.wait(5000)
        self._set_live_button(False)
        self._finish_recording(worker)
        self._set_squelch_tag(None)
        self.squelch.reset()
        fs = self._live_fs
        capture = self.live_ring.snapshot() if self.live_ring is not None else np.zeros(0, dtype=np.complex64)
        self._live_ui(False)
        self._reset_titles()
        if capture.size >= 4096:
            name = f"live_{self.live_freq.value():.3f}MHz_{datetime.now():%H%M%S}.iq"
            self._adopt_source(IQSource.from_array(capture, fs), Path(name), fs)
            saved = f" | saved {Path(worker.recordings[-1][0]).name}" if worker.recordings else ""
            self.live_status.setText(f"Stopped - captured {capture.size:,} samples ({capture.size / fs:.2f} s){saved}")
            self.statusBar().showMessage("Live stream stopped; the captured IQ is loaded for analysis.")
        else:
            self.live_status.setText("Stopped - not enough data captured")

    # ---- hover read-outs ------------------------------------------------
    def _describe_psd(self, x, _y):
        if self.psd_data is None:
            return None
        freqs, psd = self.psd_data
        if not freqs[0] <= x <= freqs[-1]:
            return None
        p = float(np.interp(x, freqs, psd)) + DBM_OFFSET_DB
        return f"{fmt_freq(x)}\n{p:.1f} dBm/Hz", p - DBM_OFFSET_DB

    def _describe_wf(self, x, y):
        if self.wf_data is None:
            return None
        m, f0, df, t0, dt = self.wf_data
        i, j = int((x - f0) // df), int((y - t0) // dt)
        if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]):
            return None
        value = float(m[i, j])
        if not np.isfinite(value):
            return None
        return f"{fmt_freq(x)}   t = {fmt_time(y)}\n{value + DBM_OFFSET_DB:.1f} dBm", None

    # ---- active channels (energy detection / blind channelizer) -----------------------------
    def _build_channel_panel(self):
        panel = QtWidgets.QGroupBox("ACTIVE CHANNELS")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItem("No channels detected")
        self.channel_combo.setToolTip("Pick a detected channel to select it on the waterfall: the ROI snaps to it and "
                                      "the constellation, classifier and demodulator follow.")
        self.channel_combo.activated.connect(self._channel_selected)
        v.addWidget(self.channel_combo)
        row = QtWidgets.QHBoxLayout()
        self.chan_threshold = QtWidgets.QDoubleSpinBox()
        self.chan_threshold.setRange(3.0, 25.0)
        self.chan_threshold.setValue(8.0)
        self.chan_threshold.setSingleStep(1.0)
        self.chan_threshold.setDecimals(1)
        self.chan_threshold.setPrefix("threshold ")
        self.chan_threshold.setSuffix(" dB")
        self.chan_threshold.setToolTip("Detection margin over the local noise floor. Raise it if noise is detected, "
                                       "lower it to find weak signals.")
        self.chan_threshold.setKeyboardTracking(False)
        self.chan_threshold.valueChanged.connect(self._rescan_channels)
        row.addWidget(self.chan_threshold, 1)
        self.chan_scan_btn = QtWidgets.QPushButton("Re-scan")
        self.chan_scan_btn.clicked.connect(self._rescan_channels)
        row.addWidget(self.chan_scan_btn)
        v.addLayout(row)
        self.channel_note = self._muted("Channels are found automatically when a file is analyzed.")
        v.addWidget(self.channel_note)
        for widget in (self.chan_threshold, self.chan_scan_btn):
            widget.setMinimumWidth(70)
        return panel

    @staticmethod
    def _fmt_offset(hz):
        return ("+" if hz >= 0 else "-") + fmt_freq(abs(hz))

    def _clear_channels(self):
        for item in self.channel_curves + self.channel_texts:
            self.wf_plot.removeItem(item)
        self.channel_curves, self.channel_texts, self.channels = [], [], []
        self.chan_region.hide()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem("No channels detected")
        self.channel_combo.blockSignals(False)
        self._chan_key = None

    def _draw_channels(self, found):
        for item in self.channel_curves + self.channel_texts:
            self.wf_plot.removeItem(item)
        self.channel_curves, self.channel_texts = [], []
        self.channels = found
        _, f0, df, t0, dt = self.wf_data
        t1 = t0 + dt * self.wf_data[0].shape[1]
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem(f"All channels ({len(found)} detected)" if found else "No channels detected")
        for k, c in enumerate(found, start=1):
            label = c["label"] or "?"
            self.channel_combo.addItem(f"Channel #{k} ({label} @ {self._fmt_offset(c['center_hz'])})")
            self.channel_combo.setItemData(
                k,
                f"{label} | bandwidth {fmt_freq(c['bandwidth_hz'])} | {c['f_lo'] / 1e3:+.2f} to "
                f"{c['f_hi'] / 1e3:+.2f} kHz | SNR {c['snr_db']:.1f} dB | peak {c['peak_db']:.1f} dB/Hz",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
            curve = pg.PlotCurveItem(
                [c["f_lo"], c["f_hi"], c["f_hi"], c["f_lo"], c["f_lo"]], [t0, t0, t1, t1, t0],
                pen=pg.mkPen(AMBER, width=2, style=QtCore.Qt.PenStyle.DashLine),
            )
            curve.setZValue(800)
            text = pg.TextItem(f"#{k} {label}", color=AMBER, anchor=(0, 0))
            text.setPos(c["f_lo"], t1)
            text.setZValue(801)
            self.wf_plot.addItem(curve, ignoreBounds=True)
            self.wf_plot.addItem(text, ignoreBounds=True)
            self.channel_curves.append(curve)
            self.channel_texts.append(text)
        self.channel_combo.setCurrentIndex(0)
        self.channel_combo.blockSignals(False)
        self.chan_region.hide()
        self.channel_note.setText(
            f"{len(found)} active channel(s) above a {self.chan_threshold.value():.0f} dB threshold."
            if found else "No active channels found; try a lower threshold."
        )

    def _channel_selected(self, index):
        for k, curve in enumerate(self.channel_curves, start=1):
            selected = k == index
            curve.setPen(pg.mkPen(ACCENT if selected else AMBER, width=3 if selected else 2,
                                  style=QtCore.Qt.PenStyle.SolidLine if selected else QtCore.Qt.PenStyle.DashLine))
        if index <= 0 or index > len(self.channels):  # "All channels": back to the overview
            self.chan_region.hide()
            self.psd_plot.autoRange()
            self.wf_plot.autoRange()
            if self.roi_check.isChecked():
                self.roi_check.setChecked(False)  # returns to whole-signal analysis
            return
        c = self.channels[index - 1]
        margin = max(c["bandwidth_hz"], 4 * self.sample_rate / int(self.fft_combo.currentText()))
        self.psd_plot.setXRange(c["f_lo"] - margin, c["f_hi"] + margin, padding=0)
        self.wf_plot.setXRange(c["f_lo"] - margin, c["f_hi"] + margin, padding=0)
        self.chan_region.setRegion((c["f_lo"], c["f_hi"]))
        self.chan_region.show()
        _, _, _, t0, dt = self.wf_data
        t1 = t0 + dt * self.wf_data[0].shape[1]
        self.roi.setPos([c["f_lo"], t0])  # region-change signals start the ROI analysis
        self.roi.setSize([c["bandwidth_hz"], t1 - t0])
        self.roi.show()
        self.statusBar().showMessage(
            f"Channel #{index}: {self._fmt_offset(c['center_hz'])}, bandwidth {fmt_freq(c['bandwidth_hz'])}, "
            f"SNR {c['snr_db']:.1f} dB"
        )

    # ---- waterfall ROI --------------------------------------------------
    # ---- performance metrics ----------------------------------------------
    def _build_perf_panel(self):
        panel = QtWidgets.QGroupBox("PERFORMANCE")
        v = QtWidgets.QGridLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setVerticalSpacing(4)
        self.perf_values = {}
        for row, (key, name) in enumerate((
            ("exec", "Execution time"),
            ("msps", "Throughput"),
            ("cpu", "CPU usage"),
            ("buf", "Buffer memory"),
            ("rss", "Process memory"),
        )):
            v.addWidget(self._muted(name), row, 0)
            value = QtWidgets.QLabel("--")
            value.setObjectName("ledText")
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            v.addWidget(value, row, 1)
            self.perf_values[key] = value
        self.perf_values["exec"].setToolTip("Wall time of the last operation (plots, ROI, demodulation, audio)")
        self.perf_values["msps"].setToolTip("Samples processed by the last operation per second, in millions")
        self.perf_values["cpu"].setToolTip("This process, averaged over the last second, as a share of all cores")
        self.perf_values["buf"].setToolTip("Sample, spectrum, bit and audio buffers held by the app "
                                           "(the memory-mapped file is not counted)")
        self.perf_timer = QtCore.QTimer(self)
        self.perf_timer.setInterval(1000)
        self.perf_timer.timeout.connect(self._refresh_resource_stats)
        self.perf_timer.start()
        self._refresh_resource_stats()
        return panel

    def _record_perf(self, name, elapsed):
        ms = elapsed * 1e3
        self.perf_values["exec"].setText(f"{ms:,.1f} ms")
        self.perf_values["exec"].setToolTip(f"{name}: wall time of the last operation")
        if self._op_samples and elapsed > 0:
            self.perf_values["msps"].setText(f"{self._op_samples / elapsed / 1e6:,.2f} MSps")
        else:
            self.perf_values["msps"].setText("--")
        if not self.live_active:  # the 1 s timer keeps CPU/memory fresh while streaming
            self._refresh_resource_stats()

    def buffer_bytes(self):
        arrays = [self.iq, self.center, self.roi_iq, *self.stage_bits.values()]
        if self.wf_data is not None:
            arrays.append(self.wf_data[0])
        if self.psd_data is not None:
            arrays += list(self.psd_data)
        return sum(a.nbytes for a in arrays if a is not None) + len(self.audio_pcm)

    def _refresh_resource_stats(self):
        self.perf_values["cpu"].setText(f"{self.cpu_meter.sample():.1f} %")
        self.perf_values["buf"].setText(format_bytes(self.buffer_bytes()))
        rss = process_rss_bytes()
        self.perf_values["rss"].setText("n/a" if rss is None else format_bytes(rss))

    # ---- audio demodulation player ------------------------------------------
    def _build_audio_panel(self):
        panel = QtWidgets.QGroupBox("AUDIO DEMOD PLAYER")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)
        v.addWidget(self._muted("Mode"))
        self.audio_mode = QtWidgets.QComboBox()
        self.audio_mode.addItems(list(AUDIO_MODES))
        self.audio_mode.setToolTip("AM envelope, narrow FM or wide FM (with 75 us de-emphasis)")
        v.addWidget(self.audio_mode)
        self.audio_secs = QtWidgets.QSpinBox()
        self.audio_secs.setRange(1, AUDIO_MAX_SECONDS)
        self.audio_secs.setValue(5)
        self.audio_secs.setPrefix("listen ")
        self.audio_secs.setSuffix(" s")
        v.addWidget(self.audio_secs)
        v.addWidget(self._muted("Plays the waterfall ROI band when ROI mode is on, otherwise the whole band "
                                "from the start of the file."))
        self.audio_btn = QtWidgets.QPushButton("Play")
        self.audio_btn.setObjectName("primary")
        self.audio_btn.clicked.connect(self.toggle_audio)
        v.addWidget(self.audio_btn)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._muted("Volume"))
        self.audio_volume = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.audio_volume.setRange(0, 100)
        self.audio_volume.setValue(70)
        self.audio_volume.valueChanged.connect(self._audio_volume_changed)
        row.addWidget(self.audio_volume, 1)
        v.addLayout(row)
        self.audio_status = self._muted("")
        v.addWidget(self.audio_status)
        if QtMultimedia is None:
            panel.setEnabled(False)
            self.audio_status.setText("QtMultimedia is not available in this PyQt6 install.")
        return panel

    def _audio_volume_changed(self, value):
        if self.audio_sink is not None:
            self.audio_sink.setVolume(value / 100.0)

    def toggle_audio(self):
        if self.audio_sink is not None and self.audio_btn.text() == "Stop":
            self.stop_audio()
        else:
            self.play_audio()

    def stop_audio(self):
        if self.audio_sink is not None:
            self.audio_sink.stateChanged.disconnect(self._audio_state)
            self.audio_sink.stop()
            self.audio_sink = None
        if self.audio_buffer is not None:
            self.audio_buffer.close()
        self.audio_btn.setText("Play")

    def _audio_state(self, state):
        # pull-mode sink goes idle once the buffer has been played out
        if state in (QtMultimedia.QAudio.State.IdleState, QtMultimedia.QAudio.State.StoppedState):
            self.stop_audio()
            self.audio_status.setText("Finished.")

    def _build_roi_panel(self):
        panel = QtWidgets.QGroupBox("WATERFALL ROI")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)
        self.roi_check = QtWidgets.QCheckBox("Analyze ROI only")
        self.roi_check.setToolTip("Drag or resize the green box on the waterfall to slice that time/frequency "
                                  "region; the constellation, classifier and demodulator then use only it.")
        self.roi_check.toggled.connect(self._roi_toggled)
        v.addWidget(self.roi_check)
        self.roi_label = self._muted("Drag the green box on the waterfall.")
        v.addWidget(self.roi_label)
        self.roi_reset_btn = QtWidgets.QPushButton("Reset ROI")
        self.roi_reset_btn.clicked.connect(self._reset_roi)
        v.addWidget(self.roi_reset_btn)
        return panel

    def _default_roi(self):
        m, f0, df, t0, dt = self.wf_data
        fw, th = df * m.shape[0], dt * m.shape[1]
        self._roi_silent = True
        self.roi.setPos([f0 + 0.3 * fw, t0 + 0.25 * th])
        self.roi.setSize([0.4 * fw, 0.5 * th])
        self._roi_silent = False
        self.roi.maxBounds = QtCore.QRectF(f0, t0, fw, th)
        self.roi.show()

    def _reset_roi(self):
        if self.wf_data is None:
            return
        self._default_roi()
        if self.roi_mode:
            self._apply_roi()

    def _roi_moved(self, *_):
        if getattr(self, "_roi_silent", False) or self.source is None:
            return
        if not self.roi_check.isChecked():
            self.roi_check.blockSignals(True)
            self.roi_check.setChecked(True)
            self.roi_check.blockSignals(False)
        self.roi_timer.start()  # debounce: apply once the drag pauses

    def _roi_toggled(self, on):
        if self.source is None:
            return
        if on:
            self._apply_roi()
        else:
            self.roi_mode = False
            self.roi_iq = None
            self.roi_label.setText("ROI off: analysing the whole signal.")
            self.run_analysis()

    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open signal file",
            str(self.file_path.parent) if self.file_path else "",
            "Signal files (*.iq *.IQ *.wav *.WAV);;All files (*)",
        )
        if path:
            self.load_file(path)

    @staticmethod
    def _plot_png(plot_item, width=2000):
        """Static high-resolution PNG (bytes) of a pyqtgraph plot."""
        exporter = ImageExporter(plot_item)
        exporter.parameters()["width"] = width
        image = exporter.export(toBytes=True)
        raw = QtCore.QByteArray()
        buffer = QtCore.QBuffer(raw)
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        return bytes(raw)

    def _square_plot_png(self, widget, size=1400):
        """Snapshot of a plot as a square image: the on-screen constellation is wide and short, which
        would leave the points tiny. The plot item is resized directly (this also works while its
        tab is hidden) with the aspect lock off and fixed +-1.15 axes, then everything is restored."""
        item = widget.getPlotItem()
        vb = item.vb

        def render(px):
            old_geometry, old_range = item.geometry(), vb.viewRange()
            old_lock = vb.state["aspectLocked"]
            vb.setAspectLocked(False)
            item.setGeometry(QtCore.QRectF(0, 0, 640, 640))
            vb.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
            item.layout.activate()
            QtWidgets.QApplication.processEvents()
            try:
                return self._plot_png(item, px)
            finally:
                item.setGeometry(old_geometry)
                vb.setRange(xRange=old_range[0], yRange=old_range[1], padding=0)
                vb.setAspectLocked(old_lock if old_lock else True)

        render(200)  # warm-up: the axes only lay out correctly on the second resize
        return render(size)

    def export_results(self):
        if self.iq is None:
            return
        default = str(self.file_path.with_name(self.file_path.stem + "_results.json"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export results (writes .json and .csv)", default, "JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        summary = {
            "file": self.file_path.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modulation": self.detected_label,
            "confidence_percent": None if self.confidence is None else round(self.confidence, 2),
            "demodulator": self.scheme_used,
            "sample_rate_hz": self.sample_rate,
            "baud_rate": self.symbol_rate,
            "snr_db": None if self.snr_db is None else round(self.snr_db, 2),
            "sync_marker": self.sync_used,
            "stage": self.stage_combo.currentText() or None,
            "evm_rms_pct": None if self.constellation_metrics is None else round(self.constellation_metrics["evm_rms_pct"], 2),
            "evm_snr_db": None if self.constellation_metrics is None else round(self.constellation_metrics["snr_db"], 2),
            "phase_noise_deg": None if self.constellation_metrics is None else round(self.constellation_metrics["phase_jitter_deg"], 2),
            "phase_offset_deg": None if self.constellation_metrics is None else round(self.constellation_metrics["phase_offset_deg"], 2),
            "descrambler": self._post_options()["descramble_scheme"],
            "bits_inverted": self.post_invert_check.isChecked(),
            "frames": frames_to_records(self.frames),
        }
        try:
            jp, cp = write_results(path, summary)
        except OSError as exc:
            self._error("Export failed", exc)
            return
        self.statusBar().showMessage(f"Exported {jp.name} and {cp.name}")

    def _current_rate(self, required):
        text = self.rate_edit.text().strip()
        if not text:
            if required:
                raise ValueError("Enter a sample rate.")
            return None
        try:
            return parse_rate(text)
        except ValueError:
            raise ValueError(f"Cannot parse sample rate {text!r}.")

    def _show_cfo(self):
        self.cfo_value.setText("n/a" if self.cfo_hz is None else fmt_freq(self.cfo_hz))

    # ---- background tasks ----------------------------------------------------------------------
    # Heavy work runs in Task threads (see the module-level *_job functions); results come back as
    # signals and are drawn here, on the GUI thread. One "main" task at a time (load / analysis /
    # ROI / channel scan / audio / PDF); the bit-stream correlation has its own latest-wins task.

    @property
    def busy(self):
        return self._main_task is not None

    def _start_task(self, name, fn, args, handlers=None, on_done=None, on_fail=None, main=True, message=None,
                    slot=None):
        task = Task(name, fn, args, handlers, on_done, on_fail, parent=self)
        task.progress.connect(self._on_task_progress)
        task.done.connect(self._on_task_done)
        task.failed.connect(self._on_task_failed)
        task.finished.connect(self._on_thread_finished)
        self._threads.add(task)
        slot = slot or ("main" if main else "corr")
        if slot == "main":
            self._main_task = task
        elif slot == "live":
            self._live_task = task
        else:
            self._corr_task = task
        if message:
            self.statusBar().showMessage(message)
        self._update_controls()
        task.start()
        return task

    def _on_thread_finished(self):
        task = self.sender()
        self._threads.discard(task)
        if task is not None:
            task.deleteLater()

    def _finish_task(self, task):
        if task is self._main_task:
            self._main_task = None
        if task is self._corr_task:
            self._corr_task = None
        if task is self._live_task:
            self._live_task = None

    def _on_task_progress(self, task, kind, payload):
        handler = task.handlers.get(kind)
        if handler is not None:
            handler(payload)

    def _on_task_done(self, task, result):
        self._finish_task(task)
        if task.name not in ("bit_view", "live_bits"):
            self._op_samples = result.get("samples", 0) if isinstance(result, dict) else 0
            self._record_perf(task.name, task.elapsed)
        self._update_controls()
        if task.on_done is not None:
            task.on_done(result)
        if self._roi_pending and not self.busy:
            self._apply_roi()

    def _on_task_failed(self, task, exc):
        self._finish_task(task)
        self._update_controls()
        if task.on_fail is not None:
            task.on_fail(exc)
        else:
            self._error(f"{task.name} failed", exc)
        if self._roi_pending and not self.busy:
            self._apply_roi()

    def wait_for_tasks(self, timeout_ms=120000):
        """Process events until every background task has finished (used on exit and by tests)."""
        end = time.perf_counter() + timeout_ms / 1e3
        while (self._main_task is not None or self._corr_task is not None or self._live_task is not None
               or self._threads) and time.perf_counter() < end:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            QtCore.QThread.msleep(5)

    def _update_controls(self):
        """Enable each control from three facts: a signal is loaded, a live stream runs, a task runs."""
        busy, live, has_file = self.busy, self.live_active, self.source is not None
        for w in (self.live_backend, self.live_rate):
            w.setEnabled(not live)
        self.open_btn.setEnabled(not live and not busy)
        for w in (self.run_btn, self.auto_btn, self.pdf_btn):
            w.setEnabled(has_file and not live and not busy)
        self.export_btn.setEnabled(has_file and not live)
        for w in (self.roi_check, self.roi_reset_btn, self.channel_combo, self.chan_threshold, self.chan_scan_btn):
            w.setEnabled(not live)
        self.audio_btn.setEnabled(not live and (not busy or self.audio_btn.text() == "Stop"))
        self.live_btn.setEnabled(not busy or live)
        self.set_engine_state(busy or self._corr_task is not None)

    def _pipeline_opts(self, symrate_text=None):
        """Snapshot of the baseband settings for a worker (widgets are never read off-thread)."""
        il_spec = INTERLEAVERS[self.interleave_combo.currentText()]
        return {
            "symrate_text": self.symrate_edit.text().strip() if symrate_text is None else symrate_text,
            "scheme": self.demod_combo.currentText(),
            "interleaver": None if il_spec is None else (il_spec[0], self.il_a_spin.value(), self.il_b_spin.value()),
            "fec": FEC_DECODERS[self.fec_combo.currentText()],
            "skip": self.skip_spin.value(),
            "invert": self.invert_check.isChecked(),
        }

    def _roi_params(self):
        pos, size = self.roi.pos(), self.roi.size()
        f_lo, f_hi = sorted((pos.x(), pos.x() + size.x()))
        t_lo, t_hi = sorted((pos.y(), pos.y() + size.y()))
        return {"f_lo": f_lo, "f_hi": f_hi, "t_lo": t_lo, "t_hi": t_hi}

    # ---- loading ------------------------------------------------------------------------------
    def load_file(self, path):
        path = Path(path)
        if self.busy:
            self.statusBar().showMessage("Busy: wait for the current task to finish.")
            return
        try:
            fs = self._current_rate(required=False)
        except ValueError as exc:
            self._error("Invalid sample rate", exc)
            return
        try:
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"'{path.name}' is empty (0 bytes).")
            if path.suffix.lower() not in (".iq", ".wav"):
                raise ValueError(f"Unsupported file type '{path.suffix or '(none)'}': expected .iq or .wav.")
        except Exception as exc:
            self._file_load_error(path, exc)
            return
        ctx = {"path": path, "is_wav": path.suffix.lower() == ".wav", "fs": fs, "dtype": self.format_combo.currentText()}
        self._start_task("load", load_job, (ctx,), on_done=self._loaded,
                         on_fail=lambda exc, p=path: self._file_load_error(p, exc),
                         message=f"Loading {path.name}...")

    def _file_load_error(self, path, exc):
        """Corrupted, zero-byte, unreadable or non-IQ/WAV file: report it cleanly, never crash."""
        self.statusBar().showMessage(f"Invalid file: {path.name}: {exc}")
        QtWidgets.QMessageBox.critical(
            self, "Invalid File Format",
            f"Could not open '{path.name}':\n\n{exc}",
        )

    def _loaded(self, result):
        self._adopt_source(result["source"], result["path"], result["fs"], iq=result["iq"])

    def _adopt_source(self, source, path, fs, iq=None):
        """Make `source` (a file or a captured live buffer) the current signal and analyse it."""
        if self.source is not None:
            self.source.close()
        self.source = source
        self.file_path = path
        self.iq = iq if iq is not None else source.read(0, WINDOW_SAMPLES)
        self.sample_rate = fs
        self.rate_edit.setText(f"{fs:g}")
        self.file_label.setText(path.name)
        self.symbol_rate = None
        self.scheme_used = None
        self.stage_bits = {}
        self._show_frames([], None)
        self._chan_key = None
        self._start_analysis(pipeline=False, auto=False)

    # ---- analysis / auto-analysis / ROI ---------------------------------------------------------
    def _start_analysis(self, pipeline, auto):
        if self.source is None:
            return False
        key = (self.file_path, self.fft_combo.currentText(), self.chan_threshold.value())
        ctx = {
            "source": self.source, "fs": self.sample_rate, "nfft": int(self.fft_combo.currentText()),
            "classifier": self.classifier, "classifier_error": self.classifier_error, "iq": self.iq,
            "pipeline": pipeline, "auto": auto, "opts": self._pipeline_opts(),
            "roi": self._roi_params() if self.roi_mode else None,
            "scan_channels": key != self._chan_key, "threshold": self.chan_threshold.value(),
            "name": self.file_path.name,
        }
        self._pending_chan_key = key
        handlers = {"spectra": self._apply_spectra, "signal": self._apply_signal, "channels": self._apply_channels,
                    "roi": self._apply_roi_result, "roi_error": self._roi_failed, "auto_scheme": self._auto_scheme,
                    "pipeline": self._apply_pipeline}
        return self._start_task("auto_analysis" if auto else "analysis", analysis_job, (ctx,), handlers,
                                on_done=self._analysis_done,
                                message="Auto-analyzing..." if auto else "Analyzing...")

    def run_analysis(self):
        if self.iq is None:
            return
        if self.busy:
            self.statusBar().showMessage("Busy: wait for the current task to finish.")
            return
        try:
            self.sample_rate = self._current_rate(required=True)
        except ValueError as exc:
            self._error("Invalid sample rate", exc)
            return
        self._start_analysis(pipeline=True, auto=False)

    def auto_analyze(self):
        """Classify -> pick demodulator -> estimate baud -> demodulate/de-interleave/FEC -> sync search."""
        if self.iq is None:
            return
        if self.busy:
            self.statusBar().showMessage("Busy: wait for the current task to finish.")
            return
        try:
            self.sample_rate = self._current_rate(required=True)
        except ValueError as exc:
            self._error("Invalid sample rate", exc)
            return
        self._auto_report = None
        self._start_analysis(pipeline=True, auto=True)

    def _analysis_done(self, result):
        if result.get("auto_fail"):
            self._auto_fail(result["auto_fail"])
        elif not result.get("pipeline_ran") and not result.get("roi_failed") and result.get("auto") is False \
                and self.file_path is not None:
            self.statusBar().showMessage(f"Analyzed {self.file_path.name}")

    def _auto_fail(self, reason):
        self._auto_report = None
        self.statusBar().showMessage(f"Auto-Analysis stopped: {reason}")

    def _auto_scheme(self, label):
        self.demod_combo.setCurrentText(label)
        self.symrate_edit.clear()  # blank -> the pipeline estimates and fills it in
        self.tabs.setCurrentIndex(1)
        self._auto_report = {"label": label}

    def _apply_roi(self):
        if self.source is None:
            return
        if self.busy:
            self._roi_pending = True  # run again with the latest box as soon as the task ends
            return
        self._roi_pending = False
        ctx = {"source": self.source, "fs": self.sample_rate, "roi": self._roi_params(), "classifier": self.classifier,
               "opts": self._pipeline_opts()}
        self._start_task("roi", roi_job, (ctx,), {"roi": self._apply_roi_result, "roi_error": self._roi_failed,
                                                  "pipeline": self._apply_pipeline}, message="Analyzing ROI...")

    def _rescan_channels(self, *_):
        if self.source is None or self.live_active or self.psd_data is None:
            return
        if self.busy:
            self.statusBar().showMessage("Busy: wait for the current task to finish.")
            return
        ctx = {"source": self.source, "fs": self.sample_rate, "psd_data": self.psd_data,
               "threshold": self.chan_threshold.value(), "classifier": self.classifier}
        self._pending_chan_key = (self.file_path, self.fft_combo.currentText(), self.chan_threshold.value())
        self._start_task("channels", channels_job, (ctx,), on_done=self._channels_done, message="Scanning channels...")

    def _channels_done(self, result):
        if "error" in result:
            self._clear_channels()
            self.channel_note.setText(result["error"])
        else:
            self._apply_channels(result)

    # ---- drawing results (GUI thread) --------------------------------------------------------------
    def _apply_spectra(self, r):
        fs, wf = self.sample_rate, r["wf"]
        freqs, psd, wf_freqs, wf_times = r["freqs"], r["psd"], r["wf_freqs"], r["wf_times"]
        self.center = r["center"]
        self.psd_curve.setData(freqs, psd)
        self.psd_plot.autoRange()
        self.psd_data = (freqs, psd)

        self.wf_image.setImage(wf, autoLevels=False)
        df = wf_freqs[1] - wf_freqs[0] if wf_freqs.size > 1 else fs
        dt = wf_times[1] - wf_times[0] if wf_times.size > 1 else r["n"] / fs
        t_start = max(0.0, float(wf_times[0] - dt / 2))  # rows are time-slice centres; the image starts at the slice edge
        self.wf_image.setRect(QtCore.QRectF(wf_freqs[0] - df / 2, t_start, df * wf.shape[0], dt * wf.shape[1]))
        new_file = self.wf_data is None or self.wf_data[0].shape != wf.shape or self._roi_file != self.file_path
        self.wf_data = (wf, float(wf_freqs[0] - df / 2), float(df), t_start, float(dt))
        if new_file:
            self._roi_file = self.file_path
            self.roi_mode, self.roi_iq = False, None
            self.roi_check.blockSignals(True)
            self.roi_check.setChecked(False)
            self.roi_check.blockSignals(False)
            self.roi_label.setText("Drag the green box on the waterfall.")
            self._default_roi()
        lo, hi = np.percentile(wf, [5, 99.5])
        self.wf_hist.setLevels(float(lo), float(hi))
        self.wf_plot.autoRange()
        n = r["n"]
        self.info_label.setText(
            f"{n:,} samples" + (" (memory-mapped)" if self.source.is_memmap else "") + chr(10)
            + f"{fs / 1e6:.4g} MS/s" + chr(10) + f"{n / fs:.4g} s"
        )

    def _show_label(self, info):
        """Modulation / confidence read-outs; keeps the previous label if classification failed."""
        if self.classifier is None:
            self.mod_value.setText("Model unavailable")
            self.conf_value.setText("--")
            self.statusBar().showMessage(self.classifier_error or "Classifier not loaded")
        elif info.get("classify_error"):
            self.mod_value.setText("Error")
            self.conf_value.setText("--")
            self.statusBar().showMessage(f"Classification failed: {info['classify_error']}")
        elif info["label"] is not None:
            self.detected_label, self.confidence = info["label"], info["confidence"]
            self.mod_value.setText(info["label"])
            self.conf_value.setText(f"{info['confidence']:.1f}%")

    def _apply_signal(self, info):
        self.rate_value.setText(f"{self.sample_rate / 1e6:.4g} MS/s")
        self.snr_db, self.snr_method = info["snr_db"], info["snr_method"]
        self.snr_value.setText(f"{self.snr_db:.1f} dB")
        self.snr_value.setToolTip(SNR_METHOD_NOTES[self.snr_method])
        self.snr_bar.set_snr(self.snr_db)
        suffix = " (clipped)" if self.snr_method == "clipped" else " (no signal found)" if self.snr_method == "default" else ""
        self.snr_quality.setText(f"{self.snr_db:.1f} dB - {QualityBar.quality(self.snr_db)}{suffix}")
        self._show_label(info)
        self.cfo_hz = info["cfo_hz"]
        self._show_cfo()
        self.const_scatter.setData(info["ci"], info["cq"])
        self._clear_metrics()

    def _apply_channels(self, result):
        self._chan_key = self._pending_chan_key
        self._draw_channels(result["found"])

    def _apply_roi_result(self, roi):
        self.roi_mode, self.roi_iq, self.roi_fs = True, roi["y"], roi["fs"]
        if not self.roi_check.isChecked():
            self.roi_check.blockSignals(True)
            self.roi_check.setChecked(True)
            self.roi_check.blockSignals(False)
        self.roi_label.setText(roi["note"])
        self._show_label(roi)
        self.cfo_hz = roi["cfo_hz"]
        self._show_cfo()
        self.const_scatter.setData(roi["ci"], roi["cq"])
        self._clear_metrics()
        self.statusBar().showMessage(roi["note"])

    def _roi_failed(self, message):
        self.roi_label.setText(message)
        self.statusBar().showMessage(f"ROI: {message}")

    def _apply_metrics(self, m):
        if "note" in m:
            self._clear_metrics(m["note"])
            return
        metrics = m["metrics"]
        self.constellation_metrics = metrics
        evm = metrics["evm_rms_pct"]
        color = GREEN if evm < 8 else AMBER if evm < 15 else RED
        sep = f' <span style="color:{MUTED}">|</span> '
        self.metrics_label.setText(
            f'EVM: <span style="color:{color}">{evm:.1f}% RMS</span>{sep}'
            f'SNR: {metrics["snr_db"]:.1f} dB{sep}'
            f'Phase Noise: {metrics["phase_jitter_deg"]:.1f}°{sep}'
            f'Offset: {metrics["phase_offset_deg"]:+.1f}°'
        )
        self.const_scatter.setData(m["pts"].real, m["pts"].imag)
        self.ideal_scatter.setData(m["ideal"].real, m["ideal"].imag)

    def _apply_pipeline(self, res):
        self.stage_bits = {}
        if "error" in res:
            self.stage_combo.blockSignals(True)
            self.stage_combo.clear()
            self.stage_combo.blockSignals(False)
            self.bin_view.setPlainText("")
            self.hex_view.setPlainText("")
            self._show_frames([], None)
            self._clear_metrics("EVM: n/a | demodulation failed")
            self.bit_summary.setText(f"Bit extraction failed: {res['error']}")
            self.statusBar().showMessage(f"Bit extraction failed: {res['error']}")
            if self._auto_report is not None:
                self._auto_fail(res["error"])
            return
        if res["estimated"] and res.get("fill_rate", True):
            self.symrate_edit.setText(f"{res['symbol_rate']:.6g}")
        self.stage_bits = res["stages"]
        self.symbol_rate = res["symbol_rate"]
        self.scheme_used = res["scheme"]
        self.bit_notes = res["bit_notes"]
        self._apply_metrics(res["metrics"])
        self.stage_combo.blockSignals(True)
        self.stage_combo.clear()
        self.stage_combo.addItems(list(self.stage_bits))
        self.stage_combo.setCurrentIndex(self.stage_combo.count() - 1)
        self.stage_combo.blockSignals(False)
        self._refresh_bit_view()

    # ---- bit stream view (correlation runs in its own worker, latest request wins) -----------------
    def _refresh_bit_view(self, *_):
        bits = self.stage_bits.get(self.stage_combo.currentText())
        if bits is None:
            return
        self._corr_seq += 1
        if self._corr_task is not None:
            self._corr_dirty = True  # re-run with the newest settings when the running one ends
            return
        opts = {"markers": self.sync_edit.text(), "max_errors": self.sync_err_spin.value(),
                "post": self._post_options(), "mode": self.render_combo.currentText()}
        task = self._start_task("bit_view", bit_view_job, (bits, opts), on_done=self._apply_bit_view, main=False)
        task.seq = self._corr_seq

    def _apply_bit_view(self, res):
        if self._corr_dirty or self.stage_bits.get(self.stage_combo.currentText()) is None:
            self._corr_dirty = False
            self._refresh_bit_view()
            return
        n_bits = res["n_bits"]
        shown = min(n_bits, VIEW_MAX_BYTES * 8)
        note = f"  |  {'  |  '.join(self.bit_notes)}" if self.bit_notes else ""
        trunc = f" (showing first {shown:,})" if shown < n_bits else ""
        self._bit_summary_base = f"{n_bits:,} bits / {n_bits // 8:,} bytes{trunc}{note}"
        self.bit_summary.setText(self._bit_summary_base)
        self._update_bit_summary_note()
        self.bin_view.setPlainText(res["bin_text"])
        self.hex_view.setPlainText(res["hex_text"])
        self.frames, self.sync_used = res["frames"], res["marker"]
        self.set_lock(bool(self.frames) and self.sync_used is not None, self.sync_used)
        self.header_view.setPlainText(res["header_text"])
        self.payload_view.setPlainText(res["payload_text"])
        if self._auto_report is not None:
            label = self._auto_report["label"]
            self._auto_report = None
            n = sum(len(f.payload) for f in self.frames)
            found = f"{len(self.frames)} frame(s) via {self.sync_used}" if self.frames else "no sync marker found"
            self.statusBar().showMessage(
                f"Auto-Analysis Complete: Detected {label} at {self.symbol_rate:,.0f} Baud  ({found}, {n} payload bytes)"
            )

    def _show_frames(self, frames, marker, error=None):
        """Synchronous refresh of the frame viewers (used to clear them; results arrive via _apply_bit_view)."""
        self.frames = frames
        self.sync_used = marker
        self.set_lock(bool(frames) and marker is not None, marker)
        header, payload = render_frames_text(frames, marker, error, self.render_combo.currentText(),
                                             self._post_options(), bool(self.stage_bits))
        self.header_view.setPlainText(header)
        self.payload_view.setPlainText(payload)

    # ---- audio ----------------------------------------------------------------------------------
    def play_audio(self):
        if self.source is None:
            self.audio_status.setText("Open a file first.")
            return
        if self.busy:
            self.audio_status.setText("Busy: wait for the current task to finish.")
            return
        self.stop_audio()
        fs = self.sample_rate
        max_count = min(int(self.audio_secs.value() * fs), AUDIO_MAX_SAMPLES)
        if self.roi_mode:
            roi = self._roi_params()
            f_lo, f_hi = roi["f_lo"], roi["f_hi"]
            start, count = int(max(roi["t_lo"], 0) * fs), min(int((roi["t_hi"] - roi["t_lo"]) * fs), max_count)
            where = f"ROI {fmt_freq(f_lo)} to {fmt_freq(f_hi)}"
        else:
            start, count, f_lo, f_hi, where = 0, max_count, -fs / 2, fs / 2, "whole band"
        ctx = {"source": self.source, "fs": fs, "f_lo": f_lo, "f_hi": f_hi, "start": start, "count": count,
               "mode": self.audio_mode.currentText(), "where": where}
        self.audio_status.setText("Demodulating audio...")
        self._start_task("audio", audio_job, (ctx,), on_done=self._start_playback,
                         on_fail=lambda exc: (self.audio_status.setText(f"Audio failed: {exc}"),
                                              self.statusBar().showMessage(f"Audio failed: {exc}")))

    def _start_playback(self, result):
        pcm, rate, note = result["pcm"], result["rate"], result["note"]
        try:
            device = QtMultimedia.QMediaDevices.defaultAudioOutput()
            if device.isNull():
                raise ValueError("No audio output device found.")
            fmt = QtMultimedia.QAudioFormat()
            fmt.setSampleRate(int(rate))
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QtMultimedia.QAudioFormat.SampleFormat.Int16)
            if not device.isFormatSupported(fmt):
                raise ValueError(f"{device.description()} does not support {int(rate)} Hz mono audio.")
        except (ValueError, OSError) as exc:
            self.audio_status.setText(f"Audio failed: {exc}")
            self.statusBar().showMessage(f"Audio failed: {exc}")
            return
        self.audio_pcm = pcm
        self.audio_bytes = QtCore.QByteArray(pcm)
        self.audio_buffer = QtCore.QBuffer(self.audio_bytes, self)
        self.audio_buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        self.audio_sink = QtMultimedia.QAudioSink(device, fmt, self)
        self.audio_sink.setVolume(self.audio_volume.value() / 100.0)
        self.audio_sink.stateChanged.connect(self._audio_state)
        self.audio_sink.start(self.audio_buffer)
        self.audio_btn.setText("Stop")
        self.audio_status.setText(f"Playing: {note}")

    def set_engine_state(self, busy):
        live = getattr(self, "live_active", False)
        color, text = (AMBER, "PROCESSING") if busy else (GREEN, "STREAMING") if live else ("#475569", "IDLE")
        self.engine_led.set_color(color)
        self.engine_text.setText(text)
        self.engine_text.setStyleSheet(f"color: {color if (busy or live) else MUTED};")

    def _live_ui(self, active):
        self._update_controls()

    def closeEvent(self, event):
        if self.live_active:
            self.stop_live()
        self.wait_for_tasks()  # let background work finish before the window (and its data) goes away
        super().closeEvent(event)

    def export_pdf(self):
        if self.iq is None or self.file_path is None or self.busy:
            return
        default = str(self.file_path.with_name(self.file_path.stem + "_report.pdf"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PDF report", default, "PDF (*.pdf)")
        if not path:
            return
        post = self._post_options()
        applied = []
        if post["descramble_scheme"]:
            applied.append(f"descrambled with {post['descramble_scheme']} "
                           f"({'whole stream' if post['scope'] == 'stream' else 'after sync marker'})")
        if post["invert"]:
            applied.append("bits inverted")
        data = {
            "file_name": self.file_path.name,
            "sample_rate_hz": self.sample_rate,
            "snr_db": self.snr_db,
            "baud_rate": self.symbol_rate,
            "analysis_region": self.roi_label.text() if self.roi_mode else None,
            "modulation": self.detected_label,
            "confidence_percent": self.confidence,
            "demodulator": self.scheme_used,
            "evm": self.constellation_metrics,
            "sync_marker": self.sync_used,
            "stage": self.stage_combo.currentText() or None,
            "post_processing": ", ".join(applied),
            "frames": frames_to_records(self.frames),
            "bit_notes": list(self.bit_notes),
        }
        try:  # the plot snapshots must be taken here: pyqtgraph items belong to the GUI thread
            data["psd_png"] = self._plot_png(self.psd_plot.getPlotItem())
            data["constellation_png"] = self._square_plot_png(self.const_plot)
        except Exception as exc:
            self._error("PDF export failed", exc)
            return
        # laying out and writing the document happens in the background
        self._start_task("pdf", pdf_job, (path, data),
                         on_done=lambda r: self.statusBar().showMessage(f"Exported PDF report: {Path(path).name}"),
                         on_fail=lambda exc: self._error("PDF export failed", exc),
                         message="Generating PDF report...")

    def _error(self, title, exc):
        self.statusBar().showMessage(f"{title}: {exc}")
        QtWidgets.QMessageBox.warning(self, title, str(exc))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
