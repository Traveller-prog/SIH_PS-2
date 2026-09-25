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
    """
    iq = np.asarray(iq).ravel()
    if iq.size < frame_len:
        iq = np.pad(iq, (0, frame_len - iq.size))
    n = iq.size // frame_len
    iq = iq[: n * frame_len].reshape(n, frame_len)
    return np.stack([iq.real, iq.imag], axis=1).astype(np.float32)


def normalize_frames(x, eps=1e-12):
    """Scale each (2, L) frame to unit average power so amplitude does not matter."""
    power = (x ** 2).sum(dim=1).mean(dim=1).clamp_min(eps).sqrt()
    return x / power[:, None, None]


class ModulationClassifier:
    """Inference wrapper: loads trained weights and predicts modulation labels."""

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
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_proba(self, x):
        """Class probabilities, shape (B, num_classes).

        x may be a complex array (1D -> one signal, 2D -> batch of signals, each
        split into 1024-sample frames whose probabilities are averaged), or a real
        array/tensor of shape (B, 2, 1024) or (2, 1024).
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        is_complex = np.iscomplexobj(x)
        if is_complex:
            signals = x[None] if x.ndim == 1 else x
            groups = [iq_to_frames(s) for s in signals]
        else:
            x = x.astype(np.float32)
            if x.ndim == 2:
                x = x[None]
            if x.ndim != 3 or x.shape[1] != 2:
                raise ValueError(f"Expected real input of shape (B, 2, L), got {x.shape}")
            groups = [x]

        probs = []
        for frames in groups:
            t = normalize_frames(torch.from_numpy(frames).to(self.device))
            p = torch.softmax(self.model(t), dim=1).cpu()
            probs.append(p.mean(dim=0, keepdim=True) if is_complex else p)
        return torch.cat(probs, dim=0).numpy()

    def predict(self, x):
        """Return a list of (label, confidence_percent) tuples, one per input."""
        probs = self.predict_proba(x)
        idx = probs.argmax(axis=1)
        return [(self.classes[i], float(probs[n, i] * 100.0)) for n, i in enumerate(idx)]
