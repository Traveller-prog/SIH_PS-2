# NTRO Signal Analyzer — NTRO PS ID 26147

A desktop application for analysing recorded radio signals. It loads raw IQ or WAV captures, identifies the
modulation with a neural network, recovers the symbol stream, undoes interleaving and forward error correction,
and searches the result for sync markers and frames.

## Features

- **File input** — raw `.iq` (int16 or float32, interleaved I/Q) and `.wav` (stereo = I/Q, mono = real). Raw files
  are memory-mapped, so large recordings open instantly and only the slices in use are read.
- **Spectrum views** — Welch power spectral density, an interactive waterfall (time–frequency) and an IQ
  constellation plot, in a dark-mode UI.
- **Modulation classification** — a 1D ResNet classifies 2-FSK, 4-FSK, BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM and
  Noise/Unknown, with a confidence score.
- **Automatic estimation** — symbol rate (cyclostationary features), carrier frequency offset (M-th power
  method) and SNR (spectral separation; an M2M4 estimator is also available in the DSP module).
- **Demodulation** — 2-FSK/4-FSK (quadrature discriminator), BPSK/QPSK (Costas loop and Gardner timing
  recovery), 16-QAM/64-QAM (slicer). Output is a raw bit array.
- **De-interleaving** — block (matrix transpose), convolutional (shift register) and diagonal.
- **FEC decoding** — Viterbi (constraint length up to 9) and Reed-Solomon (via `reedsolo` or `galois`).
- **Bit stream viewer** — every stage (demodulated, de-interleaved, decoded) in binary and hex.
- **Sync word correlation and frame parsing** — search for markers such as `0x1ACFFC1D`, `0xEB90`, `0x7EA5`
  (with a bit-error tolerance and inverted-polarity matching); split frames into header fields
  (length, frame ID, address) and payload, shown as ASCII and a hex dump.
- **One-Click Auto-Analyze** — classify, pick the demodulator, estimate the baud rate, run the whole baseband
  chain and search for sync markers in one step.
- **Protocol presets** — starting points for Marine AIS, APCO P25 and STANAG HF telemetry.
- **Live SDR + recording** — RTL-SDR / SoapySDR (or a built-in simulated source) streams into the PSD and waterfall;
  **Record Stream** appends every buffer to `recordings/*.iq` (float32 I/Q, sample rate in the file name).
- **Export** — analysis summary (modulation, baud rate, SNR, sync marker, extracted payloads) to JSON and CSV, or a
  formatted PDF report with the spectrum and constellation snapshots (`utils/pdf_exporter.py`, needs `reportlab`).

## Architecture

```
ntro-signal-analyzer/
├── gui/
│   └── main_window.py     PyQt6 + pyqtgraph application; wires every stage together
├── dsp/
│   ├── file_reader.py     .iq / .wav readers, memory-mapped IQSource
│   ├── spectral.py        PSD, STFT, constellation, symbol-rate / CFO / SNR estimation
│   ├── demodulators.py    FSK, PSK (Costas + Gardner) and QAM demodulators -> bits
│   └── correlation.py     sync-word search, frame parsing, payload export
├── fec/
│   ├── deinterleave.py    block, convolutional and diagonal de-interleavers
│   └── decoders.py        Viterbi and Reed-Solomon decoders
└── ml/
    ├── model.py           AMCNet (1D ResNet) and the ModulationClassifier wrapper
    ├── train.py           synthetic data generation and training script
    └── weights/
        └── amc_model.pth  trained weights
```

Processing chain:

```
.iq / .wav ─► IQSource (memmap) ─► PSD / waterfall / constellation
                    │
                    ├─► ModulationClassifier ─► demodulation scheme
                    ├─► symbol-rate + CFO estimation
                    ▼
        demodulator ─► de-interleaver ─► FEC decoder ─► sync search ─► header / payload ─► JSON / CSV
```

## Prerequisites

- Python 3.10 or newer (developed on 3.13)
- Windows, Linux or macOS with a display (the GUI needs one)
- Python packages: `numpy`, `scipy`, `torch`, `PyQt6`, `pyqtgraph`, `reedsolo`, `reportlab` (`galois` is optional)

## Installation

```bash
git clone <repository-url>
cd ntro-signal-analyzer

python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

pip install -r requirements.txt
```

A CPU-only PyTorch build is enough. The trained weights are included at `ml/weights/amc_model.pth`. To retrain
them (roughly 30,000 synthetic training frames, 15 epochs):

```bash
python ml/train.py
```

## Usage

1. Start the application from the repository root:

   ```bash
   python gui/main_window.py
   ```

2. Click **Open File (.iq / .wav)** and choose a recording. For `.iq` files, first select the sample format
   (int16 or float32) and enter the sample rate (for example `2.4M`); it is detected from filenames such as
   `capture_2.4Msps.iq` when possible. WAV files carry their own rate. The plots update as soon as the file loads.
3. Read the **Signal Information** panel: detected modulation, confidence, sample rate and SNR.
4. Click **One-Click Auto-Analyze** to run the full chain automatically, or set things up by hand:
   - choose a **Protocol preset** (or leave it on Custom);
   - pick the **Demodulation scheme** (or Auto to use the classifier's result);
   - enter a **Symbol rate** in baud, or leave it blank to have it estimated;
   - choose a **De-interleaver mode** and **FEC decoder** if the signal uses them;
   - enter the **Hex sync markers** to search for;
   - click **Run Analysis**.
5. Open the **Bit Stream** tab. Use the Stage dropdown to view the bits after each step, and read the extracted
   **Header Fields** and **Payload** panels. If the output is garbage, try the **Invert** checkbox or a
   **skip** offset to fix bit polarity and alignment.
6. Click **Export Results (JSON + CSV)** to save the summary and the extracted frames.

## Limitations

- The classifier was trained only on synthetic signals; accuracy on real captures is untested.
- The symbol-rate estimator can fail or be wrong on short, low-SNR bursts, and needs at least 512 samples.
- Demodulation works on the first 200,000 samples of the file, and there is no GMSK, C4FM, 8-PSK or
  differential-decoding demodulator; presets for those protocols approximate them with 2-FSK/4-FSK.
- The frame header layout is fixed (2-byte length, 2-byte frame ID, 1-byte address, big-endian).
- The protocol presets are starting points, not verified protocol implementations.
