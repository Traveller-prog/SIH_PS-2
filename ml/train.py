"""Synthetic-data training script for the modulation classifier.

Usage: python ml/train.py [--epochs 15] [--train-per-class 3750] [--val-per-class 750]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.model import CLASSES, DEFAULT_WEIGHTS, FRAME_LEN, AMCNet, normalize_frames

PSK_ORDER = {"BPSK": 2, "QPSK": 4, "8-PSK": 8}
QAM_ORDER = {"16-QAM": 16, "64-QAM": 64}
FSK_ORDER = {"2-FSK": 2, "4-FSK": 4}
SNR_RANGE_DB = (0.0, 20.0)
MAX_CFO = 0.02  # cycles/sample (fraction of sample rate)


def rrc_filter(sps, beta=0.35, span=8):
    n = np.arange(-span * sps, span * sps + 1) / sps
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


def constellation(name):
    if name in PSK_ORDER:
        m = PSK_ORDER[name]
        offset = np.pi / 4 if name == "QPSK" else 0.0
        return np.exp(1j * (2 * np.pi * np.arange(m) / m + offset))
    m = QAM_ORDER[name]
    k = int(np.sqrt(m))
    levels = np.arange(-(k - 1), k, 2)
    pts = (levels[:, None] + 1j * levels[None, :]).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def linear_symbols(name, sps, rng):
    """Pulse-shaped PSK/QAM baseband, long enough to crop a random frame from."""
    n_sym = FRAME_LEN // sps + 40
    pts = constellation(name)
    syms = pts[rng.integers(0, len(pts), n_sym)]
    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = syms
    return np.convolve(up, rrc_filter(sps), mode="same")


def fsk_signal(name, sps, rng):
    """Continuous-phase FSK baseband."""
    m = FSK_ORDER[name]
    n_sym = FRAME_LEN // sps + 40
    h = rng.uniform(0.6, 1.2)
    levels = (2 * np.arange(m) - (m - 1)) * h / (2 * sps)
    freq = np.repeat(levels[rng.integers(0, m, n_sym)], sps)
    return np.exp(1j * 2 * np.pi * np.cumsum(freq))


def generate_frame(label, rng):
    if label == "Noise/Unknown":
        sig = np.zeros(FRAME_LEN, dtype=np.complex128)
        snr_db = None
    else:
        sps = int(rng.choice([4, 6, 8, 10]))
        sig = fsk_signal(label, sps, rng) if label in FSK_ORDER else linear_symbols(label, sps, rng)
        start = rng.integers(0, len(sig) - FRAME_LEN + 1)
        sig = sig[start : start + FRAME_LEN]
        sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
        snr_db = rng.uniform(*SNR_RANGE_DB)

        cfo = rng.uniform(-MAX_CFO, MAX_CFO)
        phase = rng.uniform(0, 2 * np.pi)
        sig = sig * np.exp(1j * (2 * np.pi * cfo * np.arange(FRAME_LEN) + phase))

    noise_var = 1.0 if snr_db is None else 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_var / 2) * (rng.standard_normal(FRAME_LEN) + 1j * rng.standard_normal(FRAME_LEN))
    return sig + noise


def make_dataset(per_class, rng):
    x = np.empty((per_class * len(CLASSES), 2, FRAME_LEN), dtype=np.float32)
    y = np.empty(per_class * len(CLASSES), dtype=np.int64)
    i = 0
    for idx, label in enumerate(CLASSES):
        for _ in range(per_class):
            frame = generate_frame(label, rng)
            x[i, 0], x[i, 1] = frame.real, frame.imag
            y[i] = idx
            i += 1
    x = normalize_frames(torch.from_numpy(x))
    return TensorDataset(x, torch.from_numpy(y))


def evaluate(model, loader, loss_fn, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += loss_fn(logits, yb).item() * yb.numel()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.numel()
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train-per-class", type=int, default=3750)
    parser.add_argument("--val-per-class", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    train_ds = make_dataset(args.train_per_class, rng)
    val_ds = make_dataset(args.val_per_class, rng)
    print(f"Generated {len(train_ds)} train / {len(val_ds)} val frames in {time.time() - t0:.1f}s on {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = AMCNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        saved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            saved = "  [saved best]"
        print(
            f"Epoch {epoch}/{args.epochs}  loss {running / len(train_ds):.4f}  "
            f"val loss {val_loss:.4f}  val acc {val_acc * 100:.1f}%  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}{saved}"
        )

    print(f"Best val acc {best_acc * 100:.1f}% -> weights saved to {args.output}")


if __name__ == "__main__":
    main()
