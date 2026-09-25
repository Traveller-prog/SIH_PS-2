"""Bit/symbol de-interleavers: block matrix transpose, convolutional (Forney) and diagonal.

Each deinterleave_* function has a matching interleave_* used by the transmit side
(handy for round-trip tests). All functions take and return 1-D NumPy arrays.

Block-type functions process only whole blocks; a trailing partial block is passed
through unchanged so no data is lost.
"""

import numpy as np


def _apply_blocks(bits, block_len, perm):
    """out_block[i] = in_block[perm[i]] for each whole block; trailing remainder kept as-is."""
    bits = np.asarray(bits)
    n_blocks = len(bits) // block_len
    head = bits[: n_blocks * block_len].reshape(n_blocks, block_len)[:, perm].ravel()
    return np.concatenate([head, bits[n_blocks * block_len :]])


def _check_dims(rows, cols):
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")


def _block_perm(rows, cols):
    """Interleaver write row-wise, read column-wise: out[k] = in[perm[k]]."""
    return np.arange(rows * cols).reshape(rows, cols).T.ravel()


def _diagonal_perm(rows, cols):
    """Write row-wise, read along diagonals: for d in 0..cols-1, r in 0..rows-1 -> (r, (r+d) % cols)."""
    r = np.tile(np.arange(rows), cols)
    d = np.repeat(np.arange(cols), rows)
    return r * cols + (r + d) % cols


def interleave_block(bits, rows, cols):
    _check_dims(rows, cols)
    return _apply_blocks(bits, rows * cols, _block_perm(rows, cols))


def deinterleave_block(bits, rows, cols):
    """Block matrix-transpose deinterleaver.

    The transmitter wrote a rows x cols matrix row by row and sent it column by column;
    this writes column by column and reads row by row.
    """
    _check_dims(rows, cols)
    return _apply_blocks(bits, rows * cols, np.argsort(_block_perm(rows, cols)))


def interleave_diagonal(bits, rows, cols):
    _check_dims(rows, cols)
    return _apply_blocks(bits, rows * cols, _diagonal_perm(rows, cols))


def deinterleave_diagonal(bits, rows, cols):
    """Diagonal deinterleaver (inverse of interleave_diagonal), block size rows * cols."""
    _check_dims(rows, cols)
    return _apply_blocks(bits, rows * cols, np.argsort(_diagonal_perm(rows, cols)))


def _delay_branches(bits, branches, delay_of):
    """Delay branch i (samples i, i+B, i+2B, ...) by delay_of(i) branch-symbols, zero filled."""
    bits = np.asarray(bits)
    out = np.zeros_like(bits)
    for i in range(branches):
        d = delay_of(i)
        branch = bits[i::branches]
        shifted = np.zeros_like(branch)
        if d < len(branch):
            shifted[d:] = branch[: len(branch) - d]
        out[i::branches] = shifted
    return out


def interleave_convolutional(bits, branches, delay):
    """Forney convolutional interleaver: branch i has a shift register of i*delay cells."""
    if branches < 1 or delay < 0:
        raise ValueError("branches must be >= 1 and delay >= 0")
    return _delay_branches(bits, branches, lambda i: i * delay)


def deinterleave_convolutional(bits, branches, delay, compensate=True):
    """Convolutional (shift-register) deinterleaver.

    Branch i has (branches - 1 - i) * delay cells, complementing the interleaver.
    The end-to-end latency is branches * (branches - 1) * delay bits; with
    compensate=True the first that many output bits (start-up fill) are dropped so
    the result lines up with the original stream, and the stream is shortened
    accordingly. Feed extra flush bits at the end if the tail matters.
    """
    if branches < 1 or delay < 0:
        raise ValueError("branches must be >= 1 and delay >= 0")
    out = _delay_branches(bits, branches, lambda i: (branches - 1 - i) * delay)
    latency = branches * (branches - 1) * delay
    return out[latency:] if compensate else out
