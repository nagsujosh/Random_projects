#!/usr/bin/env python3
"""
Turn image-based PDFs into clean text/markdown.

Features:
- Tesseract OCR (lang: --lang)
- Preprocessing: deskew, denoise, binarize, contrast boost
- Reflow paragraphs, fix hyphenated line breaks
- Remove repeated headers/footers + page numbers
- Outputs .txt or .md (and optional debug JSON)

Usage examples:
  python ocr_readings.py input.pdf --out out.md
  python ocr_readings.py input.pdf --out out.txt --lang eng --dpi 300
  python ocr_readings.py input.pdf --out out.md --debug-json debug.json

Install dependencies:
    pip install pytesseract Pillow pymupdf opencv-python unidecode pandas numpy
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from unidecode import unidecode

# -----------------------------
# Helpers
# -----------------------------

def pil_from_pix(pix) -> Image.Image:
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)

def to_gray(np_img: np.ndarray) -> np.ndarray:
    if len(np_img.shape) == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    return np_img

def auto_deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate skew via minAreaRect on edges; rotate to deskew."""
    # Invert for text as white to help detection
    inv = cv2.bitwise_not(gray)
    # Threshold to isolate text
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Canny + dilate to get big contours
    edges = cv2.Canny(thr, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    coords = np.column_stack(np.where(edges > 0))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # minAreaRect returns angle in [-90, 0); convert to deskew angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.4:
        return gray  # already fine

    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def enhance(gray: np.ndarray) -> np.ndarray:
    """Light denoise + contrast boost + adaptive binarization."""
    # Gentle denoise
    den = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Contrast-limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(den)
    # Adaptive threshold (keeps faint text)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return thr

def render_pdf_page(pdf_page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
    return pil_from_pix(pix)

def ocr_image(pil_img: Image.Image, lang: str) -> List[Dict[str, Any]]:
    """Return Tesseract DATA rows (words) with positions."""
    df = pytesseract.image_to_data(pil_img, lang=lang, output_type=Output.DATAFRAME, config="--psm 3")
    if df is None or len(df) == 0:
        return []
    df = df.dropna(subset=["text"])
    rows = df.to_dict("records")
    # Keep only non-empty words
    rows = [r for r in rows if str(r.get("text", "")).strip() != ""]
    return rows

def group_lines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group words into lines (using block/par/line from Tesseract)."""
    lines = []
    by_line = defaultdict(list)
    for r in rows:
        key = (r["block_num"], r["par_num"], r["line_num"])
        by_line[key].append(r)
    for key in sorted(by_line.keys()):
        ws = sorted(by_line[key], key=lambda x: x["word_num"])
        text = " ".join(str(w["text"]) for w in ws)
        x1 = min(w["left"] for w in ws); y1 = min(w["top"] for w in ws)
        x2 = max(w["left"] + w["width"] for w in ws); y2 = max(w["top"] + w["height"] for w in ws)
        confs = [float(w["conf"]) for w in ws if w.get("conf") not in (None, -1)]
        lines.append({
            "text": text,
            "bbox": (x1, y1, x2, y2),
            "avg_conf": (sum(confs) / len(confs)) if confs else None,
            "block": key[0], "par": key[1], "line": key[2]
        })
    # Sort top-to-bottom
    lines.sort(key=lambda L: (L["bbox"][1], L["bbox"][0]))
    return lines

def remove_headers_footers(pages_lines: List[List[str]]) -> List[List[str]]:
    """
    Detect repeated first/last lines across pages and drop them (running headers/footers).
    Heuristic: if a line (normalized) repeats on >= 40% of pages at *top* or *bottom*, remove.
    """
    def norm(s: str) -> str:
        s = unidecode(s)
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        # drop standalone page numbers like "123", "Page 12", "12 of 300"
        s = re.sub(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", "", s, flags=re.I)
        return s.strip().lower()

    n = len(pages_lines)
    top_counts = Counter()
    bot_counts = Counter()

    for lines in pages_lines:
        if not lines:
            continue
        top = norm(lines[0]) if lines else ""
        bot = norm(lines[-1]) if lines else ""
        if top:
            top_counts[top] += 1
        if bot:
            bot_counts[bot] += 1

    top_rm = {s for s, c in top_counts.items() if c >= max(2, math.ceil(0.4 * n))}
    bot_rm = {s for s, c in bot_counts.items() if c >= max(2, math.ceil(0.4 * n))}

    cleaned = []
    for lines in pages_lines:
        if not lines:
            cleaned.append(lines)
            continue
        keep = []
        for i, line in enumerate(lines):
            ln = norm(line)
            if i == 0 and ln in top_rm:
                continue
            if i == len(lines) - 1 and ln in bot_rm:
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned

def reflow_paragraphs(lines: List[Dict[str, Any]]) -> List[str]:
    """
    Merge wrapped lines into paragraphs and fix hyphenation.
    Heuristics:
      - new paragraph if vertical gap jumps or line is very short and followed by big gap
      - join words split with hyphen at end-of-line (e.g., 'informa-' + 'tion' -> 'information')
    """
    if not lines:
        return []

    # Estimate median line spacing to detect paragraph breaks
    ys = [ln["bbox"][1] for ln in lines]
    ys_sorted = sorted(ys)
    gaps = [ys_sorted[i+1] - ys_sorted[i] for i in range(len(ys_sorted)-1)]
    med_gap = np.median(gaps) if gaps else 12

    paras = []
    cur = ""

    def append_line_text(acc: str, nxt: str) -> str:
        # Hyphenation fix: end with hyphen and next starts lowercase/alpha
        if re.search(r"[A-Za-z]-$", acc.strip()) and re.match(r"^[a-z].*", nxt.strip()):
            return re.sub(r"-$", "", acc.strip()) + nxt.strip()  # drop hyphen, no space
        # Otherwise normal space join
        if acc.endswith("-"):  # weird hyphen scenario
            acc = acc[:-1]
        if acc and not acc.endswith((" ", "\n")):
            acc += " "
        return acc + nxt.strip()

    for i, ln in enumerate(lines):
        text = ln["text"].strip()
        if not text:
            continue
        if cur == "":
            cur = text
        else:
            cur = append_line_text(cur, text)

        # Decide if we end the paragraph here
        # Look at next line vertical gap
        end_para = False
        if i < len(lines) - 1:
            y_curr_bottom = ln["bbox"][3]
            y_next_top = lines[i+1]["bbox"][1]
            gap = y_next_top - y_curr_bottom
            if gap > 1.6 * med_gap:
                end_para = True
        else:
            end_para = True

        if end_para:
            paras.append(cur.strip())
            cur = ""

    return paras

def normalize_text_for_ai(s: str) -> str:
    """Light cleanup for AI ingestion."""
    s = unidecode(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

@dataclass
class PageDebug:
    lines_raw: List[str]
    lines_after_strip: List[str]
    paragraphs: List[str]

# -----------------------------
# Main pipeline
# -----------------------------

def process_pdf(
    pdf_path: Path,
    lang: str = "eng",
    dpi: int = 300,
) -> Tuple[List[List[str]], List[PageDebug]]:
    """
    Returns:
      pages_paragraphs: list of pages, each a list of paragraphs (strings)
      debug: per-page debug info
    """
    doc = fitz.open(str(pdf_path))
    pages_line_texts: List[List[str]] = []
    page_debug: List[PageDebug] = []

    # Pass 1: OCR all pages to lines
    for i in range(len(doc)):
        page = doc.load_page(i)
        pil = render_pdf_page(page, dpi=dpi)
        np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = to_gray(np_img)
        gray = auto_deskew(gray)
        thr = enhance(gray)

        # Use binary for OCR to help clarity
        pil_bin = Image.fromarray(thr)
        rows = ocr_image(pil_bin, lang=lang)
        lines = group_lines(rows)

        # Keep only text portions with decent confidence or all if empty conf
        selected = []
        for ln in lines:
            txt = ln["text"].strip()
            if not txt:
                continue
            conf = ln.get("avg_conf")
            if conf is None or conf >= 40:  # keep conservative
                selected.append(ln)

        raw_text_lines = [ln["text"] for ln in selected]
        pages_line_texts.append(raw_text_lines)
        page_debug.append(PageDebug(lines_raw=raw_text_lines, lines_after_strip=[], paragraphs=[]))

    # Pass 2: strip repeated headers/footers across pages
    stripped_lines = remove_headers_footers(pages_line_texts)

    # Pass 3: reflow into paragraphs per page
    pages_paragraphs: List[List[str]] = []
    for i, (raw_lines, page) in enumerate(zip(stripped_lines, page_debug)):
        # We need line geometry for paragraph detection -> re-OCR lines for bbox alignment
        # Simpler approach: rebuild small line objects with increasing y index
        selected = [{"text": t, "bbox": (0, i * 20 + j * 20, 0, i * 20 + j * 20 + 10)}  # fake geometry if missing
                   for j, t in enumerate(raw_lines)]
        # paragraph reflow without geometry by splitting on blank lines if geometry is fake
        if len(selected) == 0:
            pages_paragraphs.append([])
            continue

        # If we lost geometry, fallback to simple paragraph merging by blank lines
        # Try a more robust approach: treat short lines + hyphenation as same paragraph
        paras = []
        cur = ""
        for idx, ln in enumerate(raw_lines):
            line = ln.strip()
            if not line:
                if cur:
                    paras.append(cur.strip())
                    cur = ""
                continue
            # hyphen fix
            if cur.endswith("-") and re.match(r"^[a-z].*", line):
                cur = cur[:-1] + line
            else:
                if cur and not cur.endswith(" "):
                    cur += " "
                cur += line
        if cur:
            paras.append(cur.strip())

        page.lines_after_strip = raw_lines
        page.paragraphs = paras
        pages_paragraphs.append(paras)

    return pages_paragraphs, page_debug

def write_outputs(
    pages_paragraphs: List[List[str]],
    out_path: Path,
    md: bool,
    add_page_dividers: bool,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if md:
        chunks = []
        for i, paras in enumerate(pages_paragraphs, start=1):
            if add_page_dividers:
                chunks.append(f"# Page {i}\n")
            for p in paras:
                chunks.append(p)
                chunks.append("")  # blank line between paragraphs
            if add_page_dividers:
                chunks.append("\n---\n")
        content = "\n".join(chunks).strip() + "\n"
    else:
        # Plain text
        chunks = []
        for i, paras in enumerate(pages_paragraphs, start=1):
            if add_page_dividers:
                chunks.append(f"[Page {i}]")
            chunks.extend(paras)
            if add_page_dividers:
                chunks.append("\n" + ("-" * 40) + "\n")
        content = "\n\n".join(chunks).strip() + "\n"

    content = normalize_text_for_   (content)
    out_path.write_text(content, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="OCR class readings (image-PDF) into clean text/markdown.")
    ap.add_argument("pdf", type=str, help="Input PDF path.")
    ap.add_argument("--out", type=str, required=True, help="Output file (.txt or .md).")
    ap.add_argument("--lang", type=str, default="eng", help="Tesseract languages, e.g., 'eng' or 'eng+spa'.")
    ap.add_argument("--dpi", type=int, default=300, help="Rendering DPI for PDF pages.")
    ap.add_argument("--no-page-dividers", action="store_true", help="Do not include per-page markers.")
    ap.add_argument("--debug-json", type=str, default=None, help="Optional debug JSON path.")
    args = ap.parse_args()

    in_path = Path(args.pdf)
    out_path = Path(args.out)
    md = out_path.suffix.lower() == ".md"

    if not in_path.exists():
        raise FileNotFoundError(f"PDF not found: {in_path}")
    if out_path.suffix.lower() not in (".txt", ".md"):
        raise ValueError("Output must end with .txt or .md")

    pages_paragraphs, debug = process_pdf(in_path, lang=args.lang, dpi=args.dpi)
    write_outputs(pages_paragraphs, out_path, md=md, add_page_dividers=not args.no_page_dividers)

    if args.debug_json:
        dj = {
            "source_pdf": str(in_path),
            "pages": [{
                "lines_raw": d.lines_raw,
                "lines_after_strip": d.lines_after_strip,
                "paragraphs": d.paragraphs
            } for d in debug]
        }
        Path(args.debug_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.debug_json).write_text(json.dumps(dj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] Wrote {out_path}")

if __name__ == "__main__":
    main()
