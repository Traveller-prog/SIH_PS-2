"""LFSR scramblers / descramblers for bit streams (uint8 0/1 arrays, MSB-first order).

Two families:

* Additive (synchronous): the data is XORed with a free-running PN sequence from a
  linear-feedback shift register (Fibonacci form). The descrambler needs the same
  polynomial *and* the same starting state, and must be started at the frame boundary.
      PN15   x^15 + x^14 + 1                (ITU-T O.150 PRBS-15)
      PN23   x^23 + x^18 + 1                (ITU-T O.150 PRBS-23)
      CCSDS  x^8 + x^7 + x^5 + x^3 + 1      (CCSDS 131.0-B pseudo-randomizer, all-ones seed)
* Multiplicative (self-synchronizing): each output bit is the input XORed with earlier
  *received* bits, so no seed is needed and errors last only one register length.
      G3RUH  x^17 + x^12 + 1                (9600 baud amateur packet)
      PN15 / PN23 are also offered in this form.

Additive registers: stage 1 receives the feedback, stage `degree` is the output, feedback =
XOR of the stages listed in `taps` (the CCSDS sequence starts FF 48 0E C0 9A 0D 70 BC).
Multiplicative descramblers use the polynomial exponents as delays.
"""

import numpy as np

# name -> (degree, taps)
POLYNOMIALS = {
    "PN15": (15, (15, 14)),
    "PN23": (23, (23, 18)),
    "CCSDS": (8, (8, 5, 3, 1)),  # feedback stages for h(x) = x^8+x^7+x^5+x^3+1; verified against the standard sequence
    "G3RUH": (17, (17, 12)),
}


def _lookup(poly):
    if isinstance(poly, str):
        try:
            return POLYNOMIALS[poly.upper()]
        except KeyError:
            raise ValueError(f"Unknown polynomial {poly!r}; choose from {sorted(POLYNOMIALS)}") from None
    degree, taps = poly
    if not taps or max(taps) != degree or min(taps) < 1:
        raise ValueError("taps must be stage numbers 1..degree and include the degree")
    return int(degree), tuple(int(t) for t in taps)


def pn_sequence(n_bits, poly="PN15", seed=None):
    """First n_bits of the LFSR output (uint8). seed: initial register contents as an int
    (bit 0 = stage 1); default all ones."""
    degree, taps = _lookup(poly)
    state = (1 << degree) - 1 if seed is None else int(seed) & ((1 << degree) - 1)
    if state == 0:
        raise ValueError("an all-zero LFSR state never leaves zero")
    out = np.empty(int(n_bits), dtype=np.uint8)
    top = degree - 1
    tap_bits = [t - 1 for t in taps]
    for i in range(len(out)):
        out[i] = (state >> top) & 1
        fb = 0
        for t in tap_bits:
            fb ^= (state >> t) & 1
        state = ((state << 1) | fb) & ((1 << degree) - 1)
    return out


def scramble_additive(bits, poly="PN15", seed=None):
    bits = np.asarray(bits, dtype=np.uint8)
    return bits ^ pn_sequence(len(bits), poly, seed)


descramble_additive = scramble_additive  # XOR with the same sequence is its own inverse


def descramble_multiplicative(bits, poly="G3RUH"):
    """Self-synchronizing descrambler: out[n] = in[n] ^ in[n-d] for every polynomial exponent d.
    The first `degree` outputs use an all-zero history, so they may be wrong."""
    degree, taps = _lookup(poly)
    x = np.asarray(bits, dtype=np.uint8)
    out = x.copy()
    for d in taps:
        if d < len(x):
            out[d:] ^= x[:-d]
    return out


def scramble_multiplicative(bits, poly="G3RUH"):
    """Inverse of descramble_multiplicative: out[n] = in[n] ^ out[n-d] (zero start state)."""
    degree, taps = _lookup(poly)
    x = np.asarray(bits, dtype=np.uint8)
    out = np.zeros_like(x)
    for n in range(len(x)):
        v = int(x[n])
        for d in taps:
            if n >= d:
                v ^= int(out[n - d])
        out[n] = v
    return out


# GUI / correlation names -> descrambler(bits)
SCHEMES = {
    "PN15": lambda b: descramble_additive(b, "PN15"),
    "PN23": lambda b: descramble_additive(b, "PN23"),
    "CCSDS": lambda b: descramble_additive(b, "CCSDS"),
    "PN15 (multiplicative)": lambda b: descramble_multiplicative(b, "PN15"),
    "G3RUH": lambda b: descramble_multiplicative(b, "G3RUH"),
}


def descramble(bits, scheme="PN15"):
    """Descramble bits with a named scheme from SCHEMES (or any callable bits -> bits)."""
    if callable(scheme):
        return scheme(np.asarray(bits, dtype=np.uint8))
    try:
        return SCHEMES[scheme](np.asarray(bits, dtype=np.uint8))
    except KeyError:
        raise ValueError(f"Unknown descrambler {scheme!r}; choose from {list(SCHEMES)}") from None
