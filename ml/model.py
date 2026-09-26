"""1D ResNet for modulation classification on raw IQ frames of shape (B, 2, 1024)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASSES = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "Noise/Unknown"]
FRAME_LEN = 1024
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "amc_model.pth"


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class AMCNet(nn.Module):
    """ResNet-style 1D CNN: (B, 2, 1024) -> (B, num_classes) logits."""

    def __init__(self, num_classes=len(CLASSES), widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResidualBlock(in_ch, w, stride=1 if i == 0 else 2))
            blocks.append(ResidualBlock(w, w))
            in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))))


def iq_to_frames(iq, frame_len=FRAME_LEN):
    """Complex 1D array -> float32 (N, 2, frame_len) non-overlapping frames.

    Signals shorter than frame_len are zero-padded; a trailing partial frame is dropped.
    The result is written straight into one pre-allocated float32 array (real and imaginary
    parts in a single pass, no intermediate stack/astype copies).
    """
    iq = np.asarray(iq).ravel()
    if iq.size < frame_len:
        iq = np.pad(iq, (0, frame_len - iq.size))
    n = iq.size // frame_len
    view = iq[: n * frame_len].reshape(n, frame_len)  # view, no copy
    out = np.empty((n, 2, frame_len), dtype=np.float32)
    out[:, 0] = view.real
    out[:, 1] = view.imag
    return out


def normalize_frames(x, eps=1e-12, inplace=False):
    """Scale each (2, L) frame to unit average power so amplitude does not matter.

    The per-frame power is one vector norm (no temporary x**2 tensor). inplace=True divides the
    input tensor itself, for callers that own it.
    """
    power = (torch.linalg.vector_norm(x, dim=(1, 2)) / x.shape[2] ** 0.5).clamp_min(eps)
    return x.div_(power[:, None, None]) if inplace else x / power[:, None, None]


class ModulationClassifier:
    """Inference wrapper: loads trained weights and predicts modulation labels.

    The network is put in eval mode once at construction (BatchNorm uses its running statistics,
    dropout is off), its parameters are frozen (requires_grad=False) and it is warmed up with a
    dummy pass, so predictions never track gradients, never update state and do not pay
    first-call setup costs.
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS, device=None):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = list(CLASSES)
        self.model = AMCNet(num_classes=len(self.classes))
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(device=self.device, dtype=torch.float32).eval()  # inference mode for the whole app
        self.model.requires_grad_(False)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # input size is fixed, so let cuDNN pick the fastest kernels
        self._warmup()

    @torch.inference_mode()
    def _warmup(self):
        self.model(torch.zeros(1, 2, FRAME_LEN, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def predict_proba(self, x):
        """Class probabilities, shape (B, num_classes).

        x may be a complex array (1D -> one signal, 2D -> batch of signals, each
        split into 1024-sample frames whose probabilities are averaged), or a real
        array/tensor of shape (B, 2, 1024) or (2, 1024).

        Runs under torch.no_grad() (plus inference mode below) as ONE forward pass over the frames
        of all signals, then averages each signal's frames.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        is_complex = np.iscomplexobj(x)
        if is_complex:
            signals = x[None] if x.ndim == 1 else x
            groups = [iq_to_frames(s) for s in signals]
        else:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 2:
                x = x[None]
            if x.ndim != 3 or x.shape[1] != 2:
                raise ValueError(f"Expected real input of shape (B, 2, L), got {x.shape}")
            groups = [x]

        counts = [len(g) for g in groups]
        frames = groups[0] if len(groups) == 1 else np.concatenate(groups)
        with torch.inference_mode():  # stricter than no_grad: also skips autograd version tracking
            t = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).to(self.device)
            # in place only when the frames are ours (built from complex input), never a caller's array
            t = normalize_frames(t, inplace=is_complex)
            probs = torch.softmax(self.model(t), dim=1)
            if is_complex:
                probs = torch.stack([g.mean(dim=0) for g in probs.split(counts)])
            return probs.cpu().numpy()

    def predict(self, x):
        """Return a list of (label, confidence_percent) tuples, one per input."""
        probs = self.predict_proba(x)
        idx = probs.argmax(axis=1)
        return [(self.classes[i], float(probs[n, i] * 100.0)) for n, i in enumerate(idx)]
