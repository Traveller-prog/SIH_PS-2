"""PDF analysis report (reportlab): 'NTRO SIGINT SIGNAL ANALYSIS REPORT'.

build_report() takes plain data - no GUI objects - so it can be called from scripts too.
Text uses the built-in Helvetica/Courier fonts, so payload text is shown through its
printable-ASCII rendering (non-printable and non-ASCII bytes appear as '.'); the hex dump
always shows the true bytes.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPORT_TITLE = "NTRO SIGINT SIGNAL ANALYSIS REPORT"
MAX_FRAMES = 15  # frames listed in the payload summary table
HEX_SNIPPET_BYTES = 32  # bytes of each payload shown in the hex dump snippet
ASCII_CHARS = 96  # characters of ASCII payload text shown

NAVY = colors.HexColor("#0b2545")
ACCENT = colors.HexColor("#13a4c4")
GRID = colors.HexColor("#c7d0dc")
BAND = colors.HexColor("#eef2f7")
MUTED = colors.HexColor("#5b6b7f")


def _styles():
    base = getSampleStyleSheet()
    mono = ParagraphStyle("m", parent=base["Code"], fontName="Courier", fontSize=7.2, leading=8.8, leftIndent=0)
    return {
        "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=13, textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=9, leading=11),
        "head": ParagraphStyle("hd", parent=base["Normal"], fontSize=9, leading=11, textColor=colors.white,
                               fontName="Helvetica-Bold"),
        "note": ParagraphStyle("n", parent=base["Normal"], fontSize=8, leading=10, textColor=MUTED),
        "mono": mono,
        "ascii": ParagraphStyle("a", parent=mono, fontSize=8, leading=10, wordWrap="CJK"),
    }


def _fmt(value, spec="{}", none="n/a"):
    return none if value is None else spec.format(value)


def _kv_table(rows, styles, width, header=None):
    """Two-column key/value table; `header` adds a coloured title row spanning both columns."""
    data = []
    if header:
        data.append([Paragraph(header[0], styles["head"]), Paragraph(header[1], styles["head"])])
    data += [[Paragraph(f"<b>{escape(k)}</b>", styles["cell"]), Paragraph(escape(str(v)), styles["cell"])]
             for k, v in rows]
    table = Table(data, colWidths=[width * 0.36, width * 0.64], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    first = 1 if header else 0
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    style.append(("ROWBACKGROUNDS", (0, first), (-1, -1), [colors.white, BAND]))
    table.setStyle(TableStyle(style))
    return table


def _image(png_bytes, width):
    reader = ImageReader(io.BytesIO(png_bytes))
    w, h = reader.getSize()
    return Image(io.BytesIO(png_bytes), width=width, height=width * h / w)


def _hex(data, per_line=8):
    out = []
    for i in range(0, len(data), per_line):
        chunk = data[i : i + per_line]
        out.append(f"{i:04X}  " + " ".join(f"{b:02X}" for b in chunk))
    return "\n".join(out)


def _payload_table(frames, styles, width):
    """Extracted Payload Summary: one row per frame with header fields, hex snippet and ASCII text."""
    head = ["#", "Header fields", "Hex dump (snippet)", "ASCII payload"]
    rows = [[Paragraph(h, styles["head"]) for h in head]]
    for f in frames[:MAX_FRAMES]:
        hd = f.get("header") or {}
        fields = "<br/>".join(f"{escape(k)}: {v} (0x{v:X})" for k, v in hd.items()) or "(header truncated)"
        fields += f"<br/>bit offset: {f['bit_offset']}"
        if f.get("notes"):
            fields += "<br/><i>" + escape("; ".join(f["notes"])) + "</i>"
        raw = bytes.fromhex((f.get("payload_hex") or "").replace(" ", ""))
        shown = raw[:HEX_SNIPPET_BYTES]
        hex_cell = [Preformatted(_hex(shown) if shown else "(empty payload)", styles["mono"])]
        if len(raw) > len(shown):
            hex_cell.append(Paragraph(f"... +{len(raw) - len(shown)} more bytes", styles["note"]))
        text = f.get("payload_ascii") or ""
        ascii_cell = Paragraph(escape(text[:ASCII_CHARS]) + ("..." if len(text) > ASCII_CHARS else "") or "(empty)",
                               styles["ascii"])
        rows.append([str(f["frame"]), Paragraph(fields, styles["cell"]), hex_cell, ascii_cell])
    widths = [width * x for x in (0.06, 0.27, 0.40, 0.27)]
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 1), (0, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _make_decor(generated):
    def decor(canvas, doc):
        page_w, page_h = A4
        canvas.saveState()
        canvas.setFillColor(NAVY)  # header band with the report title
        canvas.rect(0, page_h - 1.7 * cm, page_w, 1.7 * cm, stroke=0, fill=1)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, page_h - 1.78 * cm, page_w, 0.08 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(doc.leftMargin, page_h - 1.08 * cm, REPORT_TITLE)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(page_w - doc.rightMargin, page_h - 1.08 * cm, generated)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, 1.0 * cm, REPORT_TITLE)
        canvas.drawRightString(page_w - doc.rightMargin, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()
    return decor


def build_report(path, data):
    """Write the PDF report to `path`.

    data keys (all optional except file_name; missing values print as n/a):
        file_name, sample_rate_hz, modulation, confidence_percent, snr_db, baud_rate   (Signal Parameters)
        analysis_region, demodulator, evm (dict: evm_rms_pct, snr_db, phase_jitter_deg,
            phase_offset_deg), timestamp                                                (Analysis Details)
        psd_png, constellation_png (PNG bytes of the plots as displayed)
        sync_marker, stage, post_processing (text), bit_notes (list of str),
        frames (list of dicts with frame, bit_offset, header, payload_hex, payload_ascii, notes)
    """
    styles = _styles()
    generated = str(data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, title=REPORT_TITLE, author="NTRO Signal Analyzer", subject="Signal analysis",
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.6 * cm, bottomMargin=1.8 * cm,
    )
    width = A4[0] - doc.leftMargin - doc.rightMargin
    story = []

    fs, baud, conf = data.get("sample_rate_hz"), data.get("baud_rate"), data.get("confidence_percent")
    story += [Paragraph("Signal Parameters", styles["h"]), _kv_table([
        ("File Name", data.get("file_name", "n/a")),
        ("Sample Rate", "n/a" if fs is None else f"{fs:,.0f} Hz ({fs / 1e6:.4g} MS/s)"),
        ("Modulation", data.get("modulation") or "n/a"),
        ("Confidence Score", _fmt(conf, "{:.1f} %")),
        ("Calculated SNR", _fmt(data.get("snr_db"), "{:.1f} dB")),
        ("Baud Rate", "n/a" if baud is None else f"{baud:,.1f} baud"),
    ], styles, width, header=("Parameter", "Value"))]

    evm = data.get("evm") or {}
    details = [
        ("Demodulation scheme used", data.get("demodulator") or "n/a"),
        ("Analysed region", data.get("analysis_region") or "Entire signal"),
    ]
    if evm:
        details += [
            ("EVM (RMS)", f"{evm['evm_rms_pct']:.1f} %"),
            ("SNR implied by EVM", f"{evm['snr_db']:.1f} dB"),
            ("Phase noise / offset", f"{evm['phase_jitter_deg']:.1f} deg / {evm['phase_offset_deg']:+.1f} deg"),
        ]
    story += [Spacer(1, 8), _kv_table(details, styles, width, header=("Analysis detail", "Value")), Spacer(1, 3),
              Paragraph("The classifier was trained on synthetic signals; confirm its result against the spectrum "
                        "and constellation before relying on it.", styles["note"])]

    if data.get("psd_png"):
        story.append(KeepTogether([Paragraph("Power Spectral Density", styles["h"]), _image(data["psd_png"], width)]))
    if data.get("constellation_png"):
        story.append(KeepTogether([Paragraph("IQ Constellation", styles["h"]),
                                   _image(data["constellation_png"], min(width, 11 * cm))]))

    frames = data.get("frames") or []
    story.append(Paragraph("Extracted Payload Summary", styles["h"]))
    story.append(_kv_table([
        ("Sync marker", data.get("sync_marker") or "none found"),
        ("Bit stream stage", data.get("stage") or "n/a"),
        ("Post-processing", data.get("post_processing") or "none"),
        ("Frames found", str(len(frames))),
    ], styles, width))
    for note in data.get("bit_notes") or []:
        story.append(Paragraph(escape(note), styles["note"]))
    if frames:
        story += [Spacer(1, 8), _payload_table(frames, styles, width)]
        if len(frames) > MAX_FRAMES:
            story.append(Paragraph(f"... {len(frames) - MAX_FRAMES} more frames not listed.", styles["note"]))
    else:
        story.append(Spacer(1, 6))
        story.append(Paragraph("No frames were extracted. Run the analysis with a sync marker that matches the "
                               "signal, or adjust the demodulation settings.", styles["body"]))

    decor = _make_decor(generated)
    doc.build(story, onFirstPage=decor, onLaterPages=decor)
    return path
