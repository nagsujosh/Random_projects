#!/usr/bin/env python3
"""
reading_pdfs.py — Convert PDFs (text or scanned) into clean Markdown or plain text.

Core features
-------------
- Text-first extraction using PyMuPDF; OCR fallback (Tesseract) per page when needed.
- Optional "--force-ocr" to OCR every page (even if text is present).
- Preprocessing for OCR: deskew, denoise, adaptive binarization, contrast-limited equalization.
- Paragraph reflow and hyphenation fixes for wrapped lines.
- Remove repeated headers/footers and page numbers.
- Page range selection: process only a slice of the PDF.
- Output .md (Markdown) or .txt (plain text); optional debug JSON.

System requirement
------------------
- Tesseract OCR engine must be installed separately for OCR to work:
  macOS (Homebrew):   brew install tesseract
  Ubuntu/Debian:      sudo apt-get update && sudo apt-get install tesseract-ocr
  Windows:            https://github.com/UB-Mannheim/tesseract/wiki

Python deps
-----------
pip install pytesseract Pillow pymupdf opencv-python unidecode pandas numpy

Usage examples
--------------
1) Simple run (text-first, OCR fallback), all pages:
   python reading_pdfs.py input.pdf --out output.md

2) Force OCR on every page:
   python reading_pdfs.py input.pdf --out output.md --force-ocr --lang eng --dpi 300

3) Only process a page range (inclusive, 1-based):
   python reading_pdfs.py input.pdf --out output.md --start-page 3 --end-page 10

4) Write plain text, suppress page dividers, and emit debug JSON:
   python reading_pdfs.py input.pdf --out output.txt --no-page-dividers --debug-json debug.json

Arguments (CLI)
---------------
Positional:
  pdf                         Path to the input PDF.

Required:
  --out OUT                   Output filename; must end with .txt or .md

Optional:
  --lang LANG                 Tesseract language(s) (default: "eng"). Examples: "eng", "eng+spa".
  --dpi DPI                   Rendering DPI for OCR pages (default: 300).
  --force-ocr                 Ignore native PDF text and OCR every page.
  --text-min-chars N          If a page’s native text has fewer than N characters, switch to OCR (default: 60).
  --start-page P              First page to process (1-based; default: 1).
  --end-page P                Last page to process (inclusive; default: last page).
  --no-page-dividers          Do not include per-page markers in the output file.
  --debug-json PATH           Write a debug JSON file with per-page details (source, lines, paragraphs).

Notes:
- Page indices in the UI and most PDF readers start at 1; this script uses the same (1-based).
- If you set --start-page > --end-page, the script raises a ValueError.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from unidecode import unidecode


# -----------------------------
# Image / OCR helpers
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
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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
    den = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(den)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    return thr


def render_pdf_page(pdf_page: fitz.Page, dpi: int) -> Image.Image:
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
    rows = [r for r in rows if str(r.get("text", "")).strip() != ""]
    return rows


def group_lines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group OCR words into lines (using block/par/line from Tesseract)."""
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
            "block": key[0], "par": key[1], "line": key[2],
        })
    lines.sort(key=lambda L: (L["bbox"][1], L["bbox"][0]))
    return lines


# -----------------------------
# Text cleanup and paragraphing
# -----------------------------

def remove_headers_footers(pages_lines: List[List[str]]) -> List[List[str]]:
    """
    Detect repeated first/last lines across pages and drop them (running headers/footers).
    Heuristic: if a line (normalized) repeats on >= 40% of pages at top or bottom, remove it.
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


def reflow_paragraphs_from_lines(lines: List[Dict[str, Any]]) -> List[str]:
    """
    Merge wrapped lines into paragraphs and fix hyphenation using geometry gaps (OCR path).
    """
    if not lines:
        return []

    ys = [ln["bbox"][1] for ln in lines]
    ys_sorted = sorted(ys)
    gaps = [ys_sorted[i+1] - ys_sorted[i] for i in range(len(ys_sorted)-1)]
    med_gap = np.median(gaps) if gaps else 12

    paras = []
    cur = ""

    def append_line(acc: str, nxt: str) -> str:
        if re.search(r"[A-Za-z]-$", acc.strip()) and re.match(r"^[a-z].*", nxt.strip()):
            return re.sub(r"-$", "", acc.strip()) + nxt.strip()  # drop hyphen, no space
        if acc.endswith("-"):
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
            cur = append_line(cur, text)

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


def reflow_paragraphs_simple(raw_lines: List[str]) -> List[str]:
    """Fallback reflow when geometry is unavailable (e.g., text-based extraction lines)."""
    paras = []
    cur = ""
    for line in raw_lines:
        ln = line.strip()
        if not ln:
            if cur:
                paras.append(cur.strip()); cur = ""
            continue
        if cur.endswith("-") and re.match(r"^[a-z].*", ln):
            cur = cur[:-1] + ln
        else:
            if cur and not cur.endswith(" "):
                cur += " "
            cur += ln
    if cur:
        paras.append(cur.strip())
    return paras


def normalize_text(s: str) -> str:
    """Light cleanup for output (whitespace, unicode -> ASCII)."""
    s = unidecode(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# -----------------------------
# Debug structure
# -----------------------------

@dataclass
class PageDebug:
    source: str                 # "text" or "ocr"
    raw_text_lines: List[str]
    lines_after_strip: List[str]
    paragraphs: List[str]
    char_count_text: int        # number of characters from native extraction for that page


# -----------------------------
# Core pipeline
# -----------------------------

def extract_text_lines_with_pymupdf(page: fitz.Page) -> List[str]:
    """
    Extract text as ordered lines using PyMuPDF's blocks/lines/spans.
    Inserts a blank line between blocks to help paragraph separation.
    """
    lines_out: List[str] = []
    blocks = page.get_text("dict").get("blocks", [])
    for b in blocks:
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if text:
                lines_out.append(text)
        # empty separator between blocks
        if lines_out and lines_out[-1] != "":
            lines_out.append("")
    # Remove trailing blank
    while lines_out and lines_out[-1] == "":
        lines_out.pop()
    return lines_out


def should_ocr_page(native_text: Optional[str], min_chars: int) -> bool:
    """
    Decide whether to OCR this page:
    - Native text missing or very small.
    - Some PDFs have only whitespace or vector outlines; treat as no text.
    """
    if native_text is None:
        return True
    stripped = native_text.strip()
    return len(stripped) < min_chars


def process_pdf(
    pdf_path: Path,
    lang: str = "eng",
    dpi: int = 300,
    force_ocr: bool = False,
    text_min_chars: int = 60,
    start_page: int = 1,                 # 1-based inclusive
    end_page: Optional[int] = None,      # 1-based inclusive
) -> Tuple[List[List[str]], List[PageDebug]]:
    """
    Process the PDF within the selected page range.

    Returns:
      pages_paragraphs: list of pages, each a list of paragraph strings
      debug:            per-page debug info
    """
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    if num_pages == 0:
        return [], []

    # Sanitize page range (1-based inclusive)
    sp = max(1, start_page)
    ep = end_page if end_page is not None else num_pages
    if sp > ep:
        raise ValueError(f"Invalid page range: start-page ({sp}) > end-page ({ep})")
    if sp > num_pages:
        raise ValueError(f"start-page ({sp}) exceeds total pages ({num_pages})")
    if ep > num_pages:
        ep = num_pages

    pages_line_texts: List[List[str]] = []
    page_debug: List[PageDebug] = []

    # Pass 1: For each page in range, extract either native text or OCR
    for i in range(sp - 1, ep):  # convert to 0-based
        page = doc.load_page(i)

        # Try native text unless forced to OCR
        use_ocr = force_ocr
        native_text_all = ""
        native_lines: List[str] = []

        if not force_ocr:
            native_text_all = page.get_text("text") or ""
            use_ocr = should_ocr_page(native_text_all, text_min_chars)
            if not use_ocr:
                native_lines = extract_text_lines_with_pymupdf(page)

        if not use_ocr:
            # Text-based page
            raw_text_lines = native_lines
            paras = reflow_paragraphs_simple(raw_text_lines)
            pages_line_texts.append(raw_text_lines)
            page_debug.append(PageDebug(
                source="text",
                raw_text_lines=raw_text_lines,
                lines_after_strip=[], paragraphs=paras,
                char_count_text=len(native_text_all.strip()),
            ))
        else:
            # OCR path
            pil = render_pdf_page(page, dpi=dpi)
            np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            gray = to_gray(np_img)
            gray = auto_deskew(gray)
            thr = enhance(gray)
            pil_bin = Image.fromarray(thr)

            rows = ocr_image(pil_bin, lang=lang)
            lines = group_lines(rows)

            selected = []
            for ln in lines:
                txt = ln["text"].strip()
                if not txt:
                    continue
                conf = ln.get("avg_conf")
                if conf is None or conf >= 40:
                    selected.append(ln)

            raw_text_lines = [ln["text"] for ln in selected]
            paras = reflow_paragraphs_from_lines(selected)
            pages_line_texts.append(raw_text_lines)
            page_debug.append(PageDebug(
                source="ocr",
                raw_text_lines=raw_text_lines,
                lines_after_strip=[], paragraphs=paras,
                char_count_text=len(native_text_all.strip()),
            ))

    # Pass 2: strip repeated headers/footers across the extracted pages
    stripped_lines = remove_headers_footers(pages_line_texts)

    # Pass 3: finalize paragraphs per page
    pages_paragraphs: List[List[str]] = []
    for dbg, raw_lines_after_strip in zip(page_debug, stripped_lines):
        dbg.lines_after_strip = raw_lines_after_strip
        # Regardless of source, reflow after header/footer removal using the simple join
        # (geometry info is lost at this point; simple join maintains consistency).
        paras = reflow_paragraphs_simple(raw_lines_after_strip)
        dbg.paragraphs = paras
        pages_paragraphs.append(paras)

    return pages_paragraphs, page_debug


def write_outputs(
    pages_paragraphs: List[List[str]],
    out_path: Path,
    md: bool,
    add_page_dividers: bool,
    start_page: int,
):
    """
    Write the final output to .md or .txt.
    start_page is used only for labeling page numbers correctly when dividers are enabled.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if md:
        chunks = []
        for idx, paras in enumerate(pages_paragraphs):
            page_num = start_page + idx
            if add_page_dividers:
                chunks.append(f"# Page {page_num}\n")
            for p in paras:
                chunks.append(p)
                chunks.append("")  # blank line
            if add_page_dividers:
                chunks.append("\n---\n")
        content = "\n".join(chunks).strip() + "\n"
    else:
        chunks = []
        for idx, paras in enumerate(pages_paragraphs):
            page_num = start_page + idx
            if add_page_dividers:
                chunks.append(f"[Page {page_num}]")
            chunks.extend(paras)
            if add_page_dividers:
                chunks.append("\n" + ("-" * 40) + "\n")
        content = "\n\n".join(chunks).strip() + "\n"

    content = normalize_text(content)
    out_path.write_text(content, encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert PDFs (text or scanned) into clean Markdown or text, optionally over a page range."
    )
    ap.add_argument("pdf", type=str, help="Input PDF path.")
    ap.add_argument("--out", type=str, required=True, help="Output file (.txt or .md).")
    ap.add_argument("--lang", type=str, default="eng", help="Tesseract languages, e.g., 'eng' or 'eng+spa'.")
    ap.add_argument("--dpi", type=int, default=300, help="Rendering DPI for OCR pages.")
    ap.add_argument("--force-ocr", action="store_true", help="Ignore native text and OCR every page.")
    ap.add_argument("--text-min-chars", type=int, default=60,
                    help="If a page’s native text has fewer chars than this, OCR it.")
    ap.add_argument("--start-page", type=int, default=1, help="First page to process (1-based, inclusive).")
    ap.add_argument("--end-page", type=int, default=None, help="Last page to process (1-based, inclusive).")
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
    if args.start_page < 1:
        raise ValueError("--start-page must be >= 1")
    if args.end_page is not None and args.end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    pages_paragraphs, debug = process_pdf(
        in_path,
        lang=args.lang,
        dpi=args.dpi,
        force_ocr=args.force_ocr,
        text_min_chars=args.text_min_chars,
        start_page=args.start_page,
        end_page=args.end_page,
    )
    write_outputs(
        pages_paragraphs,
        out_path,
        md=md,
        add_page_dividers=not args.no_page_dividers,
        start_page=args.start_page,
    )

    if args.debug_json:
        dj = {
            "source_pdf": str(in_path),
            "start_page": args.start_page,
            "end_page": args.end_page,
            "pages": [{
                "source": d.source,
                "char_count_text": d.char_count_text,
                "lines_raw": d.raw_text_lines,
                "lines_after_strip": d.lines_after_strip,
                "paragraphs": d.paragraphs
            } for d in debug]
        }
        p = Path(args.debug_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] Wrote {out_path}")


if __name__ == "__main__":
    main()

