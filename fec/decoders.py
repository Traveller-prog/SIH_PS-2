"""Forward-error-correction decoders: Viterbi (convolutional) and Reed-Solomon.

Bit arrays are uint8 0/1, MSB first when packed into bytes. Convolutional generators
are octal, e.g. the CCSDS/NASA K=7 rate-1/2 code is (0o171, 0o133). The encoder
register holds the newest bit in its MSB.
"""

import numpy as np

_MAX_K = 9


def _parity(x):
    x = np.asarray(x)
    out = np.zeros_like(x)
    while np.any(x):
        out ^= x & 1
        x = x >> 1
    return out


def convolutional_encode(bits, constraint_length, generators, terminate=True):
    """Rate-1/n convolutional encoder (zero start state, optional zero-tail termination)."""
    k = constraint_length
    bits = np.asarray(bits, dtype=np.uint8)
    if terminate:
        bits = np.concatenate([bits, np.zeros(k - 1, dtype=np.uint8)])
    state = 0
    out = np.empty((len(bits), len(generators)), dtype=np.uint8)
    for t, u in enumerate(bits):
        reg = (int(u) << (k - 1)) | state
        for j, g in enumerate(generators):
            out[t, j] = bin(reg & g).count("1") & 1
        state = reg >> 1
    return out.ravel()


def depuncture(soft, pattern):
    """Re-insert erasures (0.0) into a punctured soft stream.

    pattern: 2-D 0/1 array of shape (n_outputs, period); 1 = transmitted. Symbols are
    ordered time-major (all outputs of step 0, then step 1, ...), as the encoder emits them.
    """
    keep = np.asarray(pattern).astype(bool).T.ravel()
    soft = np.asarray(soft, dtype=np.float64)
    n_periods = int(np.ceil(len(soft) / keep.sum()))
    full = np.zeros(n_periods * len(keep))
    idx = np.flatnonzero(np.tile(keep, n_periods))[: len(soft)]
    full[idx] = soft
    return full


def viterbi_decode(received, constraint_length, generators, soft=False, terminated=True):
    """Maximum-likelihood Viterbi decoder for rate-1/n convolutional codes, K <= 9.

    Args:
        received: Coded stream, n values per input bit. Hard bits (0/1) by default; with
            soft=True, real values where positive favours bit 0 and negative bit 1
            (e.g. demodulator outputs); 0 is an erasure.
        constraint_length: K (2..9).
        generators: n octal generator polynomials.
        soft: Treat received as soft decisions.
        terminated: Encoder was flushed to state 0 with K-1 zero tail bits; the tail is
            decoded and dropped. If False, the best final state is used.

    Returns:
        uint8 array of decoded information bits.
    """
    k = int(constraint_length)
    if not 2 <= k <= _MAX_K:
        raise ValueError(f"constraint_length must be between 2 and {_MAX_K}")
    gens = [int(g) for g in generators]
    n = len(gens)
    if n < 1 or any(g <= 0 or g >= (1 << k) for g in gens):
        raise ValueError("generators must be non-zero octal values that fit in K bits")

    rx = np.asarray(received, dtype=np.float64)
    if not soft:
        rx = 1.0 - 2.0 * rx
    steps = len(rx) // n
    if steps == 0:
        return np.zeros(0, dtype=np.uint8)
    rx = rx[: steps * n].reshape(steps, n)

    n_states = 1 << (k - 1)
    mask = n_states - 1
    s_next = np.arange(n_states)
    u = s_next >> (k - 2)  # input bit that led into each state
    prev = np.stack([((s_next << 1) & mask) | b for b in (0, 1)])  # predecessors, (2, S)
    reg = (u[None, :] << (k - 1)) | prev
    sign = np.empty((2, n_states, n))
    for j, g in enumerate(gens):
        sign[:, :, j] = 1.0 - 2.0 * _parity(reg & g)

    pm = np.full(n_states, -1e18)
    pm[0] = 0.0
    decisions = np.empty((steps, n_states), dtype=np.uint8)
    for t in range(steps):
        cand = pm[prev] + sign @ rx[t]  # (2, S)
        choice = cand[1] > cand[0]
        decisions[t] = choice
        pm = np.where(choice, cand[1], cand[0])

    state = 0 if terminated else int(np.argmax(pm))
    out = np.empty(steps, dtype=np.uint8)
    for t in range(steps - 1, -1, -1):
        out[t] = state >> (k - 2)
        state = ((state << 1) & mask) | decisions[t, state]
    return out[: max(steps - (k - 1), 0)] if terminated else out


def rs_encode(data_bits, n=255, k=223, fcr=0, prim=0x11D):
    """Systematic RS(n, k) encoder over GF(256) for whole k-byte blocks (test/tx helper)."""
    import reedsolo

    data = np.packbits(np.asarray(data_bits, dtype=np.uint8))
    if len(data) % k:
        raise ValueError("data length must be a multiple of k bytes")
    codec = reedsolo.RSCodec(n - k, nsize=n, fcr=fcr, prim=prim, generator=2)
    words = [np.frombuffer(bytes(codec.encode(bytes(data[i : i + k]))), dtype=np.uint8)
             for i in range(0, len(data), k)]
    return np.unpackbits(np.concatenate(words))


def rs_decode(bits, n=255, k=223, fcr=0, prim=0x11D, backend="reedsolo", return_info=False):
    """Reed-Solomon RS(n, k) block decoder over GF(2^8); shortened codes (n < 255) allowed.

    Args:
        bits: Codeword bits, MSB first, n bytes per codeword. A trailing partial
            codeword is ignored.
        n, k: Codeword / message length in bytes; corrects up to (n-k)//2 byte errors.
        fcr, prim: First consecutive root and primitive polynomial (defaults match the
            reedsolo/galois defaults; CCSDS and DVB use other values).
        backend: 'reedsolo' or 'galois'.
        return_info: Also return {'corrected': [...], 'failed': [...]} per codeword.

    Returns:
        uint8 array of decoded message bits (k bytes per codeword). An uncorrectable
        codeword keeps its received message bytes and is flagged in the info.
    """
    if not 0 < k < n <= 255:
        raise ValueError("need 0 < k < n <= 255")
    data = np.packbits(np.asarray(bits, dtype=np.uint8))
    n_words = len(data) // n
    corrected, failed, msgs = [], [], []

    if backend == "reedsolo":
        import reedsolo

        codec = reedsolo.RSCodec(n - k, nsize=n, fcr=fcr, prim=prim, generator=2)
        for i in range(n_words):
            word = data[i * n : (i + 1) * n]
            try:
                msg, _, errata = codec.decode(bytearray(word))
                msgs.append(np.frombuffer(bytes(msg), dtype=np.uint8))
                corrected.append(len(errata))
                failed.append(False)
            except reedsolo.ReedSolomonError:
                msgs.append(word[:k].copy())
                corrected.append(0)
                failed.append(True)
    elif backend == "galois":
        import galois

        gf = galois.GF(2 ** 8, irreducible_poly=prim)
        rs = galois.ReedSolomon(255, 255 - (n - k), c=fcr, field=gf)
        pad = 255 - n  # shortened code: implicit leading zero symbols
        for i in range(n_words):
            word = data[i * n : (i + 1) * n]
            full = gf(np.concatenate([np.zeros(pad, dtype=np.uint8), word]))
            msg, n_err = rs.decode(full, errors=True)
            if int(n_err) < 0:
                msgs.append(word[:k].copy())
                corrected.append(0)
                failed.append(True)
            else:
                msgs.append(np.asarray(msg, dtype=np.uint8)[pad:])
                corrected.append(int(n_err))
                failed.append(False)
    else:
        raise ValueError("backend must be 'reedsolo' or 'galois'")

    out = np.unpackbits(np.concatenate(msgs)) if msgs else np.zeros(0, dtype=np.uint8)
    if return_info:
        return out, {"corrected": corrected, "failed": failed}
    return out
