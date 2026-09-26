"""Bitstream correlation: sync-word search, frame parsing and payload export.

Bit arrays are uint8 0/1, MSB first when packed into bytes. Sync words can be given as
an int (with a bit length), a hex string ('0x1ACFFC1D', 'EB90'), a '0b...' string or a
bit array. The search slides the marker over the stream and counts bit disagreements
(Hamming distance), so noisy streams still match with max_errors > 0.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import oaconvolve

from fec.descrambler import SCHEMES as DESCRAMBLERS
from fec.descrambler import descramble

# Well-known markers: name -> (value, bit length)
SYNC_WORDS = {
    "CCSDS ASM 0x1ACFFC1D": (0x1ACFFC1D, 32),
    "0xEB90": (0xEB90, 16),
    "0x7EA5": (0x7EA5, 16),
}


def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits):
    """Pack bits MSB first; a trailing partial byte is zero-padded."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def parse_sync_word(word, n_bits=None):
    """Return a sync marker as a uint8 bit array (see module docstring for accepted forms)."""
    if isinstance(word, str):
        s = word.strip().lower().replace("_", "").replace(" ", "")
        if s.startswith("0b"):
            digits = np.frombuffer(s[2:].encode("ascii"), dtype=np.uint8) - ord("0")
            if digits.size == 0 or np.any(digits > 1):
                raise ValueError("binary sync word may only contain 0 and 1")
            return digits
        s = s[2:] if s.startswith("0x") else s
        if not s:
            raise ValueError("empty sync word")
        value, width = int(s, 16), 4 * len(s)
    elif isinstance(word, (int, np.integer)):
        value = int(word)
        width = n_bits or max(value.bit_length(), 1)
        width = width if n_bits else 8 * ((width + 7) // 8)
    else:
        arr = np.asarray(word, dtype=np.uint8)
        if arr.ndim != 1 or arr.size == 0 or np.any(arr > 1):
            raise ValueError("sync word must be a non-empty 1-D 0/1 array")
        return arr
    width = n_bits or width
    if value >> width:
        raise ValueError("sync word does not fit in the given bit length")
    n_bytes = (width + 7) // 8  # big-endian bytes -> unpackbits -> keep the low `width` bits
    return np.unpackbits(np.frombuffer(value.to_bytes(n_bytes, "big"), dtype=np.uint8))[n_bytes * 8 - width :]


def sliding_hamming(bits, pattern):
    """Hamming distance between pattern and every window of bits (length len(bits)-L+1)."""
    bits = np.asarray(bits, dtype=np.uint8)
    pattern = np.asarray(pattern, dtype=np.uint8)
    if len(bits) < len(pattern):
        return np.zeros(0, dtype=np.int64)
    corr = cross_correlate(bits, pattern)
    return np.rint((len(pattern) - corr) / 2).astype(np.int64)


def cross_correlate(bits, pattern):
    """Bipolar (+-1) cross-correlation, 'valid' mode. Peak value == len(pattern) at a perfect match."""
    a = 1.0 - 2.0 * np.asarray(bits, dtype=np.float32)
    b = 1.0 - 2.0 * np.asarray(pattern, dtype=np.float32)
    if len(a) < len(b):
        return np.zeros(0)
    # overlap-add with a reversed pattern = correlation; far cheaper than one full-length FFT when
    # the marker is short, and +-1 sums stay exact in float32 for any realistic marker length
    return oaconvolve(a, b[::-1], mode="valid")


@dataclass
class SyncMatch:
    position: int  # bit index of the first marker bit
    errors: int  # Hamming distance to the marker
    inverted: bool  # matched the complemented marker (180-degree phase ambiguity)
    length: int  # marker length in bits

    @property
    def end(self):
        return self.position + self.length


def find_sync(bits, sync_word, n_bits=None, max_errors=0, both_polarities=True):
    """Find every occurrence of a sync marker.

    Overlapping hits are collapsed: within one marker length only the best (fewest
    errors) match is kept. Returns SyncMatch list sorted by position.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    pat = parse_sync_word(sync_word, n_bits)
    L = len(pat)
    dist = sliding_hamming(bits, pat)
    if dist.size == 0:
        return []
    cands = [(dist, False)]
    if both_polarities:
        cands.append((L - dist, True))
    # every hit of both polarities as parallel arrays (position, errors, inverted)
    parts = [(np.flatnonzero(d <= max_errors), inv) for d, inv in cands]
    pos = np.concatenate([p for p, _ in parts])
    err = np.concatenate([d[p] for (p, _), (d, _) in zip(parts, cands)]).astype(np.int64)
    inverted = np.concatenate([np.full(p.size, inv) for p, inv in parts])
    if pos.size == 0:
        return []
    order = np.argsort(pos, kind="stable")
    pos, err, inverted = pos[order], err[order], inverted[order]
    # hits closer than one marker length belong to the same sync event; keep its best (fewest
    # errors, earliest) hit. Clusters whose hits all lie within one marker length of the first
    # are resolved for all clusters at once with a lexsort; the rare longer chains (only when
    # max_errors is so loose that random windows match) fall back to the exact greedy rule.
    cluster = np.concatenate([[0], np.cumsum(np.diff(pos) >= L)])
    first = np.concatenate([[0], np.flatnonzero(np.diff(cluster)) + 1])
    last = np.concatenate([first[1:] - 1, [pos.size - 1]])
    short = (pos[last] - pos[first]) < L
    ranked = np.lexsort((np.arange(pos.size), err, cluster))
    best_of = ranked[np.searchsorted(cluster[ranked], np.arange(first.size))]  # best hit of each cluster
    kept = [best_of[short]] + [
        first[c] + np.asarray(_greedy_collapse(pos[first[c] : last[c] + 1], err[first[c] : last[c] + 1], L), dtype=np.intp)
        for c in np.flatnonzero(~short)  # only the rare over-long chains loop
    ]
    kept = np.sort(np.concatenate(kept))  # hit indices are position-ordered
    return [SyncMatch(int(pos[k]), int(err[k]), bool(inverted[k]), L) for k in kept]


def _greedy_collapse(pos, err, L):
    """Original left-to-right rule: a hit inside the current best's window replaces it only if it has strictly fewer errors."""
    kept, best = [], -1
    for i in range(len(pos)):
        if best >= 0 and pos[i] < pos[best] + L:
            if err[i] < err[best]:
                best = i
                kept[-1] = i
            continue
        best = i
        kept.append(i)
    return kept


def find_any_sync(bits, words=None, max_errors=0, both_polarities=True):
    """Search several markers at once. words: {name: (value, n_bits) | str | array}; defaults to SYNC_WORDS.

    Returns {name: [SyncMatch, ...]} with only the markers that were found.
    """
    found = {}
    for name, spec in (words or SYNC_WORDS).items():
        m = find_sync(bits, *spec, max_errors=max_errors, both_polarities=both_polarities) \
            if isinstance(spec, tuple) else find_sync(bits, spec, max_errors=max_errors,
                                                      both_polarities=both_polarities)
        if m:
            found[name] = m
    return found


@dataclass
class FrameFormat:
    """Layout after the sync marker.

    header: ordered (field_name, n_bytes) pairs, big-endian by default.
    length_field: header field holding the payload length in bytes (None = payload runs
        to the next sync marker or the end of the stream).
    length_adjust: added to that field to get the payload length (e.g. -HeaderLen when
        the field counts the whole frame).
    trailer_bytes: bytes after the payload that are not payload (CRC/FCS); kept in Frame.trailer.
    """

    header: tuple = (("length", 2), ("frame_id", 2), ("address", 1))
    length_field: str = "length"
    length_adjust: int = 0
    trailer_bytes: int = 0
    byteorder: str = "big"

    @property
    def header_bytes(self):
        return sum(n for _, n in self.header)


@dataclass
class Frame:
    sync: SyncMatch
    header: dict
    payload: bytes
    trailer: bytes = b""
    complete: bool = True  # False when the stream ended before the whole frame arrived
    notes: list = field(default_factory=list)

    @property
    def start(self):
        return self.sync.position

    def to_ascii(self):
        return payload_to_ascii(self.payload)

    def render(self, mode="ASCII"):
        return render_payload(self.payload, mode)

    def hexdump(self):
        return hexdump(self.payload)


def parse_frames(bits, sync_word, fmt=None, n_bits=None, max_errors=0, both_polarities=True,
                 descramble_scheme=None, invert=False, scope="frame"):
    """Split a bitstream into frames at each sync marker and parse header/payload.

    Frames matched on the complemented marker have their bits flipped before parsing.
    Frames need not be byte aligned in the stream; the bytes are read from the marker's end.

    Post-processing (see fec.descrambler.SCHEMES for scheme names):
        invert: flip the frame bits after the polarity fix (header and payload).
        descramble_scheme: descramble with that scheme. scope='frame' (default) restarts the
            descrambler right after each sync marker, which stays unscrambled - the usual
            frame-synchronous case; scope='stream' descrambles the whole stream first, for
            systems whose sync marker is scrambled too.
        Inversion is applied before descrambling.
    """
    fmt = fmt or FrameFormat()
    bits = np.asarray(bits, dtype=np.uint8)
    if scope not in ("frame", "stream"):
        raise ValueError("scope must be 'frame' or 'stream'")
    if descramble_scheme and scope == "stream":
        bits = descramble(bits ^ 1 if invert else bits, descramble_scheme)
        invert = False
        descramble_scheme = None
    matches = find_sync(bits, sync_word, n_bits, max_errors, both_polarities)
    frames = []
    for i, m in enumerate(matches):
        stop = matches[i + 1].position if i + 1 < len(matches) else len(bits)
        body = bits[m.end : stop]
        if m.inverted:
            body = body ^ 1
        # next-marker boundary is only authoritative when no length field exists
        if fmt.length_field is not None and i + 1 < len(matches):
            body = bits[m.end :] ^ 1 if m.inverted else bits[m.end :]
        if invert:
            body = body ^ 1
        if descramble_scheme:
            body = descramble(body, descramble_scheme)
        data = bits_to_bytes(body[: len(body) // 8 * 8])
        notes = []
        hb = fmt.header_bytes
        if len(data) < hb:
            frames.append(Frame(m, {}, b"", b"", False, ["truncated header"]))
            continue
        header, off = {}, 0
        for name, n in fmt.header:
            header[name] = int.from_bytes(data[off : off + n], fmt.byteorder)
            off += n
        rest = data[hb:]
        if fmt.length_field is None:
            plen = max(len(rest) - fmt.trailer_bytes, 0)
        else:
            plen = header[fmt.length_field] + fmt.length_adjust
            if plen < 0:
                frames.append(Frame(m, header, b"", b"", False, ["negative payload length"]))
                continue
        complete = len(rest) >= plen + fmt.trailer_bytes
        if not complete:
            notes.append(f"truncated: {len(rest)} of {plen + fmt.trailer_bytes} bytes present")
        payload = rest[:plen]
        trailer = rest[plen : plen + fmt.trailer_bytes]
        if m.errors:
            notes.append(f"sync matched with {m.errors} bit error(s)")
        if m.inverted:
            notes.append("inverted polarity")
        frames.append(Frame(m, header, payload, trailer, complete, notes))
    return frames


def payload_to_ascii(payload, replacement="."):
    """Printable ASCII view; non-printable bytes become `replacement`."""
    return "".join(chr(b) if 32 <= b < 127 else replacement for b in bytes(payload))


PAYLOAD_MODES = ("ASCII", "UTF-8", "Raw Hex")


def render_payload(payload, mode="ASCII"):
    """Payload bytes as text: 'ASCII' (non-printable -> '.'), 'UTF-8' (invalid bytes -> U+FFFD,
    control characters -> '.') or 'Raw Hex' (hex dump with an ASCII gutter)."""
    data = bytes(payload)
    if mode == "ASCII":
        return payload_to_ascii(data)
    if mode == "UTF-8":
        text = data.decode("utf-8", errors="replace")
        return "".join(c if c.isprintable() else "." for c in text)
    if mode == "Raw Hex":
        return hexdump(data)
    raise ValueError(f"mode must be one of {PAYLOAD_MODES}")


def hexdump(payload, width=16, offset=0):
    """Classic hex dump: offset, hex bytes, ASCII gutter."""
    data = bytes(payload)
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hx = " ".join(f"{b:02X}" for b in chunk).ljust(width * 3 - 1)
        lines.append(f"{offset + i:08X}  {hx}  |{payload_to_ascii(chunk)}|")
    return "\n".join(lines)


def export_payloads(frames, path=None):
    """Render all frame payloads as text (header summary, ASCII, hex dump); optionally write to path."""
    parts = []
    for n, f in enumerate(frames):
        hdr = ", ".join(f"{k}={v} (0x{v:X})" for k, v in f.header.items()) or "-"
        flag = "" if f.complete else "  [INCOMPLETE]"
        parts.append(
            f"--- Frame {n} @ bit {f.start}{flag} ---\nHeader: {hdr}\n"
            f"Payload: {len(f.payload)} bytes\nASCII: {f.to_ascii()}\n{f.hexdump()}\n"
        )
    text = "\n".join(parts)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text
