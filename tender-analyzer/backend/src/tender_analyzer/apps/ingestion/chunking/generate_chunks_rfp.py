# procurement_generate_chunks.py
# -----------------------------------------------------------------------------
# Procurement / Tender Conditions post-processor:
# - Column-aware merge per page using provided `col` field.
# - Heading stack (0..4) from numbered titles like "1", "1.1", "1.1.1".
# - Special handling for Table of Contents pages (no splitting, no hierarchy).
# - Stable UUIDv5 chunk IDs using a document fingerprint + per-chunk key.
# - Emits renderer-compatible chunks with required keys:
#   chunk_id, type, text, metadata, page_number, doc_fingerprint, bbox, orig_size.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

# ----------------------------- config & constants ---------------------------- #

# Namespace for deterministic IDs (project-specific, static)
_PROC_NAMESPACE = uuid.UUID("8b5c8d8f-32a2-4d25-9b3f-8f91b9c6e5a1")

# Default chunk "type" (override with env CHUNK_TYPE="cv_profile" if needed)
DEFAULT_CHUNK_TYPE = os.getenv("CHUNK_TYPE", "procurement_section")

# Regexes
RE_NUM_HEADING_L1 = re.compile(r"^\s*(\d+)\s+(.+)$")                  # "1 Contracting entity"
RE_NUM_HEADING_LN = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$")        # "1.2 Name and address"
RE_TOC_TITLE      = re.compile(r"\btable of contents\b", re.I)
RE_TOC_LEADERS    = re.compile(r"\.{4,}\s*\d+$")                      # "Title ....... 12"
RE_SOFT_HYPHEN    = re.compile(r"-\n(?=[a-z])")
RE_WS             = re.compile(r"[ \t\u00A0]+")


IGNORED_LABELS = {"footer", "sideinfo"}

# ----------------------------- small utilities ------------------------------ #
# --- prepend heading context into text surface --------------------------------
def _prefix_with_headings(raw_text: str, headings: list[str | None], seg_type: str) -> str:
    """
    Build 'DocTitle — H1 — H2 — ... — <raw_text>'.
    - Skips None headings.
    - If the last heading equals the raw_text (e.g., heading chunk), avoid duplicating it.
    - Does nothing for ToC segments.
    """
    if seg_type == "toc":
        return raw_text

    parts = [h for h in (headings or []) if h]
    if not parts:
        return raw_text

    def _norm(s: str) -> str:
        import re as _re
        return _re.sub(r"\s+", " ", (s or "")).strip().lower()

    # avoid repeating the current heading as both prefix and body
    if parts and _norm(parts[-1]) == _norm(raw_text):
        parts = parts[:-1]

    return " — ".join(parts + [raw_text]) if parts else raw_text


def _nz(s: str | None) -> str:
    return (s or "").strip()

def _squeeze(s: str) -> str:
    return RE_WS.sub(" ", _nz(s))

def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = RE_SOFT_HYPHEN.sub("", s)                  # join hyphen splits
    s = "\n".join(_squeeze(line) for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _union_boxes(boxes: Iterable[List[float]]) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for b in boxes:
        if not b or len(b) != 4: 
            continue
        x0, y0, x1, y1 = map(float, b)
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    if not xs0:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))

def _page_size_from_elems(elems: List[Dict[str, Any]]) -> List[float]:
    if not elems:
        # A4-ish fallback in points
        return [595.28, 841.89]
    boxes = [e.get("bbox") for e in elems if e.get("bbox")]
    _, _, x1, y1 = _union_boxes(boxes) if boxes else (0.0, 0.0, 595.28, 841.89)
    # round up so bbox always lies within page
    return [float(ceil(max(1.0, x1))), float(ceil(max(1.0, y1)))]

def _shrink_bbox(bbox: Tuple[float, float, float, float], ratio: float = 0.02) -> List[int]:
    x0, y0, x1, y1 = bbox
    w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
    dx = min(w * max(0.0, ratio), w * 0.25); dy = min(h * max(0.0, ratio), h * 0.25)
    nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
    if nx1 <= nx0:
        midx = (x0 + x1) / 2.0; nx0, nx1 = midx - 0.5, midx + 0.5
    if ny1 <= ny0:
        midy = (y0 + y1) / 2.0; ny0, ny1 = midy - 0.5, midy + 0.5
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]

def _hash_text(text: str) -> str:
    return hashlib.sha1((_nz(text)).encode("utf-8")).hexdigest()[:12]

# --------------------------- input normalization ---------------------------- #

def _as_pages_ordered(data: Any) -> List[List[Dict[str, Any]]]:
    """
    Accepts:
      - {"pages":[{"page_number":.., "elements":[...]}, ...]}
      - [{"page_number":..,"elements":[...]}, ...]
      - [[elem,...], [elem,...], ...]
    Returns: list-of-pages where each page is a list of element dicts.
    """
    if isinstance(data, dict) and "pages" in data:
        pages = sorted(data.get("pages") or [], key=lambda p: p.get("page_number", 0))
        return [p.get("elements") or [] for p in pages]
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], list):
            return data
        if isinstance(data[0], dict) and "elements" in data[0]:
            pages = sorted(data, key=lambda p: p.get("page_number", 0))
            return [p.get("elements") or [] for p in pages]
    raise TypeError("Unsupported input JSON shape. Expect dict with 'pages', or list of pages/elements.")

# --------------------------- fingerprint & IDs ------------------------------ #

def _document_fingerprint(pages_ordered: List[List[Dict[str, Any]]], sample_pages: int = 2) -> uuid.UUID:
    texts: List[str] = []
    for elems in pages_ordered[: max(1, sample_pages)]:
        ordered = sorted(elems, key=lambda b: int(b.get("reading_order", 0)))
        for b in ordered:
            t = (b.get("text") or "").strip()
            if t:
                texts.append(t)
    sample = "\n".join(texts)[:4000] or "empty-doc"
    return uuid.uuid5(_PROC_NAMESPACE, sample)

def _chunk_uuid(doc_uuid: uuid.UUID, key: str) -> str:
    return str(uuid.uuid5(doc_uuid, key))

# -------------------------- file meta passthrough --------------------------- #

def _file_meta_from_env() -> Dict[str, str | None]:
    def nz(v: str | None) -> str | None:
        v = (v or "").strip()
        return v if v else None
    return {
        "file_id": nz(os.getenv("FILE_ID")),
        "file_name": nz(os.getenv("FILE_NAME")),
        "source_file_id": nz(os.getenv("SOURCE_FILE_ID")) or nz(os.getenv("FILE_ID")),
        "source_file_name": nz(os.getenv("SOURCE_FILE_NAME")) or nz(os.getenv("FILE_NAME")),
        "content_hash": nz(os.getenv("CONTENT_HASH")),
    }

# ----------------------------- TOC detection -------------------------------- #

def _is_toc_page(elems: List[Dict[str, Any]]) -> bool:
    texts = [(_nz(e.get("text"))).lower() for e in elems if _nz(e.get("text"))]
    if any(RE_TOC_TITLE.search(t) for t in texts):
        return True
    # Heuristic: many dotted leader lines
    leaderish = sum(1 for t in texts if RE_TOC_LEADERS.search(t))
    return leaderish >= 3

# -------------------------- heading / hierarchy ----------------------------- #

def _heading_level_and_title(line: str) -> Tuple[int | None, str | None]:
    """
    Returns (level, full_title) where level in {0..4}, or (None, None) if not heading-like.
    Level mapping:
      1-digit -> level 1; dotted '1.2' -> level 2; '1.2.3' -> level 3; etc.
    """
    if not line:
        return (None, None)
    m2 = RE_NUM_HEADING_LN.match(line)
    if m2:
        dots = m2.group(1).count(".") + 1
        level = min(4, dots)              # clamp to 4
        return (level, line.strip())
    m1 = RE_NUM_HEADING_L1.match(line)
    if m1:
        return (1, line.strip())
    return (None, None)

def _update_heading_ctx(ctx: List[str | None], level: int, title: str) -> None:
    # ctx[0] is document title; 1..4 are numeric sections
    for i in range(level + 1, 5):
        ctx[i] = None
    ctx[level] = title

# --------------------------- main build function ---------------------------- #

def build_chunks(pages_ordered_or_doc: Any) -> List[Dict[str, Any]]:
    """
    Core: column-aware post-processor for procurement-style PDFs parsed by PyMuPDF.
    Input  : list-of-pages (each page is list of elements with at least {label, text, bbox, reading_order, col})
    Output : list of chunks; each chunk has keys:
             chunk_id, type, text, metadata, page_number, doc_fingerprint, bbox, orig_size
    Notes  :
      - ToC pages are emitted as ONE chunk per column with text
        "<DocTitle> – Table of Contents – <joined column text>" (no splitting).
      - For normal pages, we split by detected headings; each segment carries heading_0..4 in metadata.
    """
    pages = _as_pages_ordered(pages_ordered_or_doc)
    doc_uuid = _document_fingerprint(pages)
    file_meta_env = _file_meta_from_env()
    file_meta_present = {k: v for k, v in file_meta_env.items() if v}
    chunks: List[Dict[str, Any]] = []

    # Pre-scan for document title (first page 'sec' or first non-empty large block)
    doc_title = None
    if pages:
        first_elems = sorted(pages[0], key=lambda b: int(b.get("reading_order", 0)))
        for e in first_elems:
            if (e.get("label") or "").lower() in {"sec"} and _nz(e.get("text")):
                doc_title = _normalize_text(e.get("text")).split("\n", 1)[0]
                break
        if not doc_title:
            for e in first_elems:
                t = _normalize_text(e.get("text"))
                if t:
                    doc_title = t.split("\n", 1)[0]
                    break
    doc_title = doc_title or "Document"

    # Build per-page sizes (best-effort)
    page_sizes = [_page_size_from_elems(elems) for elems in pages]

    # Heading context across flow
    heading_ctx: List[str | None] = [None, None, None, None, None]
    heading_ctx[0] = doc_title

    # Helper to emit a chunk
    def emit_chunk(text: str, bbox_list: List[List[float]], page_index: int,
                   seg_type: str, col_index: int | None,
                   heading_ctx_snapshot: List[str | None]) -> None:
        raw_text = _normalize_text(text)
        text_surface = _prefix_with_headings(raw_text, heading_ctx_snapshot, seg_type)

        # text_surface = _normalize_text(text)
        if not text_surface:
            return
        page_bbox_tuple = _union_boxes(bbox_list)
        page_bbox = _shrink_bbox(page_bbox_tuple, ratio=0.02)
        orig_size = page_sizes[page_index] if 0 <= page_index < len(page_sizes) else [595.28, 841.89]

        # Deterministic ID key: page:col:segtype:hash(text)
        key = f"p{page_index+1}:c{col_index if col_index is not None else 0}:{seg_type}:{_hash_text(text_surface)}"
        chunk_id = _chunk_uuid(doc_uuid, key)

        metadata = {
            "segment_type": seg_type,                 # "toc" | "heading" | "body"
            "column_index": col_index,
            "heading_0_text": heading_ctx_snapshot[0],
            "heading_1_text": heading_ctx_snapshot[1],
            "heading_2_text": heading_ctx_snapshot[2],
            "heading_3_text": heading_ctx_snapshot[3],
            "heading_4_text": heading_ctx_snapshot[4],
        }

        chunk: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "type": DEFAULT_CHUNK_TYPE,
            "text": text_surface,
            "metadata": metadata,
            "page_number": int(page_index + 1),
            "doc_fingerprint": str(doc_uuid),
            "bbox": page_bbox,
            "orig_size": [float(orig_size[0]), float(orig_size[1])],
        }
        # Attach optional file meta if present (avoid blocking upstream defaults)
        for k, v in file_meta_present.items():
            chunk[k] = v
        chunks.append(chunk)

    # ------------------------- main per-page loop --------------------------- #
    for p_idx, elems in enumerate(pages):
        # Filter out footers / constant page labels if present
        filtered = []
        for e in elems:
            lbl = (e.get("label") or "").lower()
            t = _normalize_text(e.get("text") or "")
            if not t:
                continue
            if lbl in IGNORED_LABELS:
                continue
            if t.lower().startswith("page:") and re.search(r"\bof\b", t.lower()):
                # "Page: 2 of 21" type
                continue
            filtered.append(e)

        # Group by column
        cols: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for e in filtered:
            c = int(e.get("col", 0))
            cols[c].append(e)

        # TOC page? (assessed on full page)
        is_toc = _is_toc_page(filtered)

        for col_idx, col_elems in sorted(cols.items(), key=lambda kv: kv[0]):
            # Sort by reading_order then y
            ordered = sorted(col_elems, key=lambda b: (int(b.get("reading_order", 0)), float(b.get("bbox", [0,0,0,0])[1])))

            # Merge text within this column into a single stream, but keep element boundaries for bbox union per segment
            lines: List[Tuple[str, List[float]]] = []
            for e in ordered:
                t = _normalize_text(e.get("text") or "")
                if not t:
                    continue
                lines.append((t, e.get("bbox") or [0,0,0,0]))

            if not lines:
                continue

            if is_toc:
                # One chunk per column: "<Doc> – Table of Contents – <full text>"
                full_text = " ".join(t for (t, _) in lines)
                toc_text = f"{doc_title} – Table of Contents – {full_text}"
                bbox_list = [b for (_, b) in lines]
                emit_chunk(toc_text, bbox_list, p_idx, seg_type="toc", col_index=col_idx, heading_ctx_snapshot=list(heading_ctx))
                continue

            # Not ToC: split by headings and emit body segments under heading context.
            # Assemble a simple linear buffer and split by heading regexes.
            # We will create segments by scanning lines; on heading, flush current body.
            cur_body_text: List[str] = []
            cur_body_boxes: List[List[float]] = []

            def flush_body():
                if cur_body_text:
                    emit_chunk(" ".join(cur_body_text), cur_body_boxes, p_idx, seg_type="body", col_index=col_idx, heading_ctx_snapshot=list(heading_ctx))
                    cur_body_text.clear()
                    cur_body_boxes.clear()

            for (t, b) in lines:
                level, title = _heading_level_and_title(t)
                if level is not None:
                    # heading line → flush prior body, update context, emit heading chunk
                    flush_body()
                    _update_heading_ctx(heading_ctx, level, title)
                    emit_chunk(title, [b], p_idx, seg_type="heading", col_index=col_idx, heading_ctx_snapshot=list(heading_ctx))
                else:
                    # normal body line → accumulate
                    cur_body_text.append(t)
                    cur_body_boxes.append(b)
            # tail
            flush_body()
    chunks = [c for c in chunks if c.get("metadata", {}).get("segment_type") != "heading" and c.get("metadata", {}).get("column_index") != -1]
    return chunks

# ---------------------------------- CLI ------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build procurement chunks from parsed PyMuPDF JSON.")
    parser.add_argument("-i", "--input", type=str, help="Input JSON path (doc with 'pages' or list).", required=False)
    parser.add_argument("-o", "--output", type=str, default="procurement_chunks.json", help="Output JSON file.")
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")
        data = json.loads(in_path.read_text(encoding="utf-8"))
    else:
        # Tiny inline sample resembling your pages 1–4
        data = {
            "pages": [
                {"page_number": 1, "elements": [
                    {"label": "sec", "bbox": [99,125,516,235], "text": "Tender Conditions\n\nfor Consultancy Service Agreement ... Greenland", "reading_order": 0, "col": 0},
                    {"label": "footer","bbox":[206,807,526,813],"text":"27290372.1","reading_order":1,"col":-1}
                ]},
                {"page_number": 2, "elements": [
                    {"label":"para","bbox":[99,125,527,714],"text":"Table of contents\n1 Contracting entity ..... 4\n2 General information ..... 4", "reading_order":0, "col":1},
                    {"label":"footer","bbox":[206,807,526,814],"text":"27290372.1","reading_order":3,"col":-1}
                ]},
                {"page_number": 4, "elements": [
                    {"label":"sub_sec","bbox":[99,125,510,323],"text":"1\nContracting entity\n1.1\nName and address", "reading_order":0, "col":1},
                    {"label":"para","bbox":[99,326,527,666],"text":"NunaGreen Construction Disko Bay A/S ... Greenland", "reading_order":1, "col":1},
                ]},
            ]
        }

    pages_ordered = _as_pages_ordered(data)
    chunks = build_chunks(pages_ordered)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunk(s) → {out_path}")

if __name__ == "__main__":
    main()

__all__ = ["build_chunks", "main"]
