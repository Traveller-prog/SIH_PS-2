"""Main window: file sidebar + PSD, waterfall and constellation plots."""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from dsp.file_reader import IQSource, detect_sample_rate_from_name, open_iq_memmap, read_wav_file
from dsp.spectral import (
    compute_constellation,
    compute_psd_from_source,
    compute_stft_from_source,
    correct_cfo,
    estimate_snr,
    estimate_symbol_rate,
)
from dsp.correlation import find_sync, hexdump, parse_frames, parse_sync_word, payload_to_ascii
from dsp.demodulators import SUPPORTED, demodulate
from fec.deinterleave import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
)
from fec.decoders import rs_decode, viterbi_decode
from ml.model import FRAME_LEN, ModulationClassifier

# Files are memory-mapped; each task loads only the slice it needs.
WINDOW_SAMPLES = 1_048_576  # head of the file: symbol-rate estimation and demodulation
CENTER_SAMPLES = 65_536  # centre of the file: classifier, SNR, CFO, constellation
PSD_FRAMES = 2048  # frames averaged (spread over the whole file) for the PSD
WATERFALL_FRAMES = 400  # rows in the waterfall
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


def correlate_frames(bits, markers_text, max_errors=0):
    """Search bits for each comma/space separated hex marker; parse frames for the best one.

    Returns (marker_text, frames). marker_text is None when nothing matched.
    Raises ValueError on an unparsable marker.
    """
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
    return best, parse_frames(bits, best, max_errors=max_errors)


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
                  "baud_rate", "snr_db", "sync_marker", "stage")
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


BG = "#14161a"
PANEL = "#1c1f26"
ACCENT = "#3d8bfd"
TRACE = "#4cc9f0"

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: #e6e8ee; font-size: 13px; }}
QLabel {{ background: transparent; }}
#infoPanel {{ background: {BG}; border: 1px solid #2f3440; border-radius: 8px; }}
#metricValue {{ font-size: 16px; font-weight: 600; }}
#sidebar {{ background: {PANEL}; border-right: 1px solid #2a2e38; }}
#title {{ font-size: 18px; font-weight: 600; }}
QLabel[muted="true"] {{ color: #8b91a0; }}
QLineEdit, QComboBox {{ background: {BG}; border: 1px solid #2f3440; border-radius: 6px; padding: 6px 8px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: #2a2e38; border: 1px solid #353b48; border-radius: 6px; padding: 8px 12px; }}
QPushButton:hover {{ background: #333846; }}
QPushButton:disabled {{ color: #666c7a; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #5a9dff; }}
QPushButton#primary:disabled {{ background: #2a3b57; border-color: #2a3b57; color: #7c8aa3; }}
QStatusBar {{ background: {PANEL}; color: #8b91a0; }}
QSpinBox {{ background: {BG}; border: 1px solid #2f3440; border-radius: 6px; padding: 4px 6px; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {PANEL}; color: #8b91a0; padding: 8px 18px; border: 1px solid #2a2e38; border-bottom: none; }}
QTabBar::tab:selected {{ background: {BG}; color: #e6e8ee; border-top: 2px solid {ACCENT}; }}
QPlainTextEdit {{ background: {BG}; border: 1px solid #2f3440; border-radius: 6px; font-family: Consolas, "Courier New", monospace; font-size: 12px; }}
"""


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
        self.snr_db = None
        try:
            self.classifier = ModulationClassifier()
        except Exception as exc:
            self.classifier_error = str(exc)

        pg.setConfigOptions(antialias=False, background=BG, foreground="#aab0bf")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_plots(), "Spectrum && Constellation")
        self.tabs.addTab(self._build_bit_viewer(), "Bit Stream")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Open an .iq or .wav file to begin.")

    def _build_sidebar(self):
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(300)
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
        v.setSpacing(8)

        title = QtWidgets.QLabel("Signal Analyzer")
        title.setObjectName("title")
        v.addWidget(title)
        v.addSpacing(8)

        self.open_btn = QtWidgets.QPushButton("Open File (.iq / .wav)")
        self.open_btn.clicked.connect(self.pick_file)
        v.addWidget(self.open_btn)

        self.file_label = QtWidgets.QLabel("No file selected")
        self.file_label.setProperty("muted", True)
        self.file_label.setWordWrap(True)
        v.addWidget(self.file_label)
        v.addSpacing(8)

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
        v.addSpacing(8)
        v.addWidget(self._build_baseband_panel())
        v.addSpacing(8)

        self.auto_btn = QtWidgets.QPushButton("One-Click Auto-Analyze")
        self.auto_btn.setObjectName("primary")
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

        v.addSpacing(8)
        v.addWidget(self._build_info_panel())

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setProperty("muted", True)
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)
        v.addStretch(1)
        for combo in side.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
        return side

    def _build_baseband_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("infoPanel")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(4)
        header = QtWidgets.QLabel("BASEBAND PROCESSING")
        header.setProperty("muted", True)
        v.addWidget(header)

        v.addWidget(self._muted("Protocol preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        v.addWidget(self.preset_combo)
        self.preset_note = self._muted("")
        v.addWidget(self.preset_note)

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
        self.sync_err_spin.setPrefix("max bit errors ")
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

    def _build_bit_viewer(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(12, 12, 12, 12)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Stage"))
        self.stage_combo = QtWidgets.QComboBox()
        self.stage_combo.setMinimumWidth(180)
        self.stage_combo.currentTextChanged.connect(self._refresh_bit_view)
        top.addWidget(self.stage_combo)
        self.bit_summary = QtWidgets.QLabel("Press Run Analysis to extract bits.")
        self.bit_summary.setProperty("muted", True)
        self.bit_summary.setWordWrap(True)
        top.addWidget(self.bit_summary, 1)
        v.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for title, attr in (("Binary", "bin_view"), ("Hex", "hex_view")):
            box = QtWidgets.QWidget()
            bv = QtWidgets.QVBoxLayout(box)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.addWidget(self._muted(title))
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            setattr(self, attr, edit)
            bv.addWidget(edit)
            split.addWidget(box)
        split.setSizes([650, 450])

        frame_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for title, attr in (("Header Fields", "header_view"), ("Extracted Payload (ASCII / Hex)", "payload_view")):
            box = QtWidgets.QWidget()
            bv = QtWidgets.QVBoxLayout(box)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.addWidget(self._muted(title))
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            setattr(self, attr, edit)
            bv.addWidget(edit)
            frame_split.addWidget(box)
        frame_split.setSizes([450, 650])

        rows = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        rows.addWidget(split)
        rows.addWidget(frame_split)
        rows.setSizes([350, 300])
        v.addWidget(rows, 1)
        return page

    def _build_info_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("infoPanel")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        header = QtWidgets.QLabel("SIGNAL INFORMATION")
        header.setProperty("muted", True)
        v.addWidget(header)

        def metric(name):
            v.addSpacing(4)
            v.addWidget(self._muted(name))
            value = QtWidgets.QLabel("--")
            value.setObjectName("metricValue")
            v.addWidget(value)
            return value

        self.mod_value = metric("Detected Modulation")
        self.conf_value = metric("Confidence Score")
        self.rate_value = metric("Estimated Sample Rate")
        self.snr_value = metric("Calculated SNR")
        return panel

    @staticmethod
    def _muted(text):
        label = QtWidgets.QLabel(text)
        label.setProperty("muted", True)
        label.setWordWrap(True)
        return label

    def _build_plots(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self.psd_plot = pg.PlotWidget()
        style_plot(self.psd_plot.getPlotItem(), "Power Spectral Density", "Frequency (Hz)", "Power (dB/Hz)")
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen(TRACE, width=1.2))
        splitter.addWidget(self.psd_plot)

        self.wf_layout = pg.GraphicsLayoutWidget()
        self.wf_plot = self.wf_layout.addPlot(row=0, col=0)
        style_plot(self.wf_plot, "Waterfall", "Frequency (Hz)", "Time (s)")
        self.wf_image = pg.ImageItem()
        self.wf_plot.addItem(self.wf_image)
        self.wf_plot.setMouseEnabled(x=True, y=True)
        self.wf_hist = pg.HistogramLUTItem(image=self.wf_image)
        self.wf_hist.gradient.loadPreset("viridis")
        self.wf_layout.addItem(self.wf_hist, row=0, col=1)
        splitter.addWidget(self.wf_layout)

        self.const_plot = pg.PlotWidget()
        style_plot(self.const_plot.getPlotItem(), "IQ Constellation", "I", "Q")
        self.const_plot.setAspectLocked(True)
        self.const_plot.setRange(xRange=(-1.1, 1.1), yRange=(-1.1, 1.1))
        self.const_scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(76, 201, 240, 90))
        self.const_plot.addItem(self.const_scatter)
        splitter.addWidget(self.const_plot)

        splitter.setSizes([300, 320, 320])
        return splitter

    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open signal file",
            str(self.file_path.parent) if self.file_path else "",
            "Signal files (*.iq *.IQ *.wav *.WAV);;All files (*)",
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        path = Path(path)
        is_wav = path.suffix.lower() == ".wav"
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if is_wav:
                wav_iq, fs = read_wav_file(path)
                source = IQSource.from_array(wav_iq, fs)
                self.rate_edit.setText(f"{fs:g}")
            else:
                fs = self._current_rate(required=False) or detect_sample_rate_from_name(path)
                if fs is None:
                    raise ValueError(
                        "Enter the sample rate first (it could not be detected from the filename)."
                    )
                source = open_iq_memmap(path, dtype=self.format_combo.currentText(), sample_rate=fs)
                self.rate_edit.setText(f"{fs:g}")
        except Exception as exc:
            self._error(f"Could not load {path.name}", exc)
            self.run_btn.setEnabled(self.iq is not None)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if len(source) < 2:
            source.close()
            self._error("Empty file", ValueError("No samples were read."))
            return

        if self.source is not None:
            self.source.close()
        self.source = source
        self.file_path = path
        self.iq = source.read(0, WINDOW_SAMPLES)
        self.sample_rate = fs
        self.file_label.setText(path.name)
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.auto_btn.setEnabled(True)
        self.symbol_rate = None
        self.scheme_used = None
        self.stage_bits = {}
        self._show_frames([], None)
        self.update_plots()

    def run_analysis(self):
        if self.iq is None:
            return
        try:
            self.sample_rate = self._current_rate(required=True)
        except ValueError as exc:
            self._error("Invalid sample rate", exc)
            return
        self.update_plots()
        self.run_baseband_pipeline()

    def auto_analyze(self):
        """Classify -> pick demodulator -> estimate baud -> demodulate/de-interleave/FEC -> sync search."""
        if self.iq is None:
            return
        try:
            self.sample_rate = self._current_rate(required=True)
        except ValueError as exc:
            self._error("Invalid sample rate", exc)
            return
        self.update_plots()  # runs the classifier on the loaded samples
        label = self.detected_label
        if self.classifier is None or label is None:
            self._auto_fail(self.classifier_error or "the classifier is unavailable")
            return
        if label not in SUPPORTED:
            self._auto_fail(f"detected '{label}' has no demodulator; choose a scheme manually")
            return
        self.demod_combo.setCurrentText(label)
        self.symrate_edit.clear()  # blank -> the pipeline estimates and fills it in
        self.tabs.setCurrentIndex(1)
        self.run_baseband_pipeline()
        if not self.stage_bits:
            self._auto_fail(self.bit_summary.text().removeprefix("Bit extraction failed: "))
            return
        n = sum(len(f.payload) for f in self.frames)
        found = f"{len(self.frames)} frame(s) via {self.sync_used}" if self.frames else "no sync marker found"
        self.statusBar().showMessage(
            f"Auto-Analysis Complete: Detected {label} at {self.symbol_rate:,.0f} Baud"
            f"  ({found}, {n} payload bytes)"
        )

    def _auto_fail(self, reason):
        self.statusBar().showMessage(f"Auto-Analysis stopped: {reason}")

    def run_baseband_pipeline(self):
        self.stage_bits = {}
        try:
            text = self.symrate_edit.text().strip()
            rate_note = None
            if text:
                symbol_rate = parse_rate(text)
            else:
                symbol_rate, score = estimate_symbol_rate(self.iq, self.sample_rate)
                self.symrate_edit.setText(f"{symbol_rate:.6g}")
                rate_note = f"Symbol rate: {symbol_rate:,.1f} baud (estimated, {score:.0f} dB)"
            scheme = self.demod_combo.currentText()
            if scheme.startswith("Auto"):
                scheme = self.detected_label
                if scheme not in SUPPORTED:
                    raise ValueError(
                        f"Auto-detected '{scheme}' cannot be demodulated; pick a scheme manually."
                    )
            il_spec = INTERLEAVERS[self.interleave_combo.currentText()]
            interleaver = (
                None if il_spec is None else (il_spec[0], self.il_a_spin.value(), self.il_b_spin.value())
            )
            fec = FEC_DECODERS[self.fec_combo.currentText()]

            baseband, cfo = derotate(self.iq[:DEMOD_MAX_SAMPLES], self.sample_rate, scheme)
            cfo_note = f"CFO removed: {cfo:,.1f} Hz" if cfo is not None else None
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                stages, notes = run_baseband(
                    baseband,
                    self.sample_rate,
                    scheme,
                    symbol_rate,
                    interleaver,
                    fec,
                    skip_bits=self.skip_spin.value(),
                    invert=self.invert_check.isChecked(),
                )
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as exc:
            self.stage_combo.blockSignals(True)
            self.stage_combo.clear()
            self.stage_combo.blockSignals(False)
            self.bin_view.setPlainText("")
            self.hex_view.setPlainText("")
            self._show_frames([], None)
            self.bit_summary.setText(f"Bit extraction failed: {exc}")
            self.statusBar().showMessage(f"Bit extraction failed: {exc}")
            return

        self.stage_bits = stages
        self.symbol_rate = symbol_rate
        self.scheme_used = scheme
        self.bit_notes = [f"Scheme: {scheme}"] + ([rate_note] if rate_note else []) + ([cfo_note] if cfo_note else []) + notes
        self.stage_combo.blockSignals(True)
        self.stage_combo.clear()
        self.stage_combo.addItems(list(stages))
        self.stage_combo.setCurrentIndex(self.stage_combo.count() - 1)
        self.stage_combo.blockSignals(False)
        self._refresh_bit_view()

    def _refresh_bit_view(self, *_):
        bits = self.stage_bits.get(self.stage_combo.currentText())
        if bits is None:
            return
        shown = min(len(bits), VIEW_MAX_BYTES * 8)
        note = f"  |  {'  |  '.join(self.bit_notes)}" if self.bit_notes else ""
        trunc = f" (showing first {shown:,})" if shown < len(bits) else ""
        self.bit_summary.setText(f"{len(bits):,} bits / {len(bits) // 8:,} bytes{trunc}{note}")
        self.bin_view.setPlainText(format_binary(bits))
        self.hex_view.setPlainText(format_hex(bits))
        try:
            marker, frames = correlate_frames(bits, self.sync_edit.text(), self.sync_err_spin.value())
        except ValueError as exc:
            self._show_frames([], None, str(exc))
            return
        self._show_frames(frames, marker)

    def _show_frames(self, frames, marker, error=None):
        self.frames = frames
        self.sync_used = marker
        if error:
            self.header_view.setPlainText(error)
            self.payload_view.setPlainText("")
            return
        if marker is None:
            msg = "No sync marker found." if self.stage_bits else "Press Run Analysis to search for sync markers."
            self.header_view.setPlainText(msg)
            self.payload_view.setPlainText("")
            return
        head = [f"Sync marker {marker}: {len(frames)} frame(s)"]
        body = []
        for n, f in enumerate(frames[:MAX_FRAMES_SHOWN]):
            fields = "  ".join(f"{k}={v} (0x{v:X})" for k, v in f.header.items()) or "(header truncated)"
            extra = f"  [{'; '.join(f.notes)}]" if f.notes else ""
            head.append(f"#{n}  bit {f.start}\n    {fields}{extra}")
            body.append(f"--- Frame {n}: {len(f.payload)} bytes ---\nASCII: {f.to_ascii()}\n{hexdump(f.payload)}")
        if len(frames) > MAX_FRAMES_SHOWN:
            head.append(f"... {len(frames) - MAX_FRAMES_SHOWN} more not shown")
        self.header_view.setPlainText("\n".join(head))
        self.payload_view.setPlainText("\n\n".join(body))

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

    def update_plots(self):
        src, fs = self.source, self.sample_rate
        nfft = int(self.fft_combo.currentText())
        n = len(src)
        mid = max(0, n // 2 - CENTER_SAMPLES // 2)
        self.center = src.read(mid, CENTER_SAMPLES)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            freqs, psd = compute_psd_from_source(src, fs, nperseg=nfft, max_frames=PSD_FRAMES)
            wf_freqs, wf_times, wf = compute_stft_from_source(src, fs, nperseg=nfft, max_frames=WATERFALL_FRAMES)
        except Exception as exc:
            self._error("Analysis failed", exc)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.psd_curve.setData(freqs, psd)
        self.psd_plot.autoRange()

        wf = wf.astype(np.float32)
        self.wf_image.setImage(wf, autoLevels=False)
        df = wf_freqs[1] - wf_freqs[0] if wf_freqs.size > 1 else fs
        dt = wf_times[1] - wf_times[0] if wf_times.size > 1 else n / fs
        self.wf_image.setRect(
            QtCore.QRectF(wf_freqs[0] - df / 2, wf_times[0], df * wf.shape[0], dt * wf.shape[1])
        )
        lo, hi = np.percentile(wf, [5, 99.5])
        self.wf_hist.setLevels(float(lo), float(hi))
        self.wf_plot.autoRange()

        self.update_signal_info()  # classify first: the CFO order depends on the detected modulation
        corrected, self.cfo_hz = derotate(self.center, fs, self.detected_label)
        ci, cq = compute_constellation(corrected)
        self.const_scatter.setData(ci, cq)

        self.info_label.setText(
            f"{n:,} samples" + (" (memory-mapped)" if src.is_memmap else "") + chr(10) + f"{fs / 1e6:.4g} MS/s" + chr(10) + f"{n / fs:.4g} s"
        )
        self.statusBar().showMessage(f"Analyzed {self.file_path.name}")

    def update_signal_info(self):
        self.rate_value.setText(f"{self.sample_rate / 1e6:.4g} MS/s")
        try:
            self.snr_db = estimate_snr(self.center, self.sample_rate)
            self.snr_value.setText(f"{self.snr_db:.1f} dB")
        except ValueError:
            self.snr_db = None
            self.snr_value.setText("n/a")
        if self.classifier is None:
            self.mod_value.setText("Model unavailable")
            self.conf_value.setText("--")
            self.statusBar().showMessage(self.classifier_error or "Classifier not loaded")
            return
        start = max(0, self.center.size // 2 - FRAME_LEN // 2)
        frame = self.center[start : start + FRAME_LEN]
        try:
            label, conf = self.classifier.predict(frame)[0]
        except Exception as exc:
            self.mod_value.setText("Error")
            self.conf_value.setText("--")
            self.statusBar().showMessage(f"Classification failed: {exc}")
            return
        self.detected_label = label
        self.confidence = conf
        self.mod_value.setText(label)
        self.conf_value.setText(f"{conf:.1f}%")

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
