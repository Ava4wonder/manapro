# brochure_generate_chunks.py
# -----------------------------------------------------------------------------
# Brochure post-processor:
# - One chunk per page (whole-page content).
# - Qdrant-friendly, stable UUIDv5 chunk IDs.
# - Generalized "brochure" chunk schema with headings, body, captions, images.
# - Adds per-chunk `bbox` slightly smaller than the page union bbox.
# - Callable discovered by classify_type_and_postprocess via `build_chunks`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Any, Tuple

# ----------------------------- small utilities ------------------------------ #

def _squeeze(s: str) -> str:
    """Normalize whitespace and strip."""
    return re.sub(r"[ \t\u00A0]+", " ", (s or "").strip())

def _normalize_lines(s: str) -> str:
    """Collapse excessive newlines while preserving paragraph breaks."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(_squeeze(line) for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _is_heading_label(lbl: str) -> bool:
    return (lbl or "").lower() in {"sec", "sub_sec", "sub_sub_sec"}

def _is_caption_label(lbl: str) -> bool:
    return (lbl or "").lower() in {"cap", "caption"}

def _is_paragraph_label(lbl: str) -> bool:
    return (lbl or "").lower() in {"para"}

def _is_image_label(lbl: str) -> bool:
    return (lbl or "").lower() in {"fig", "image"}

_BULLET_RE = re.compile(r"^\s*(?:[\-\u2022\u2023\u25E6\u2043\u2219]|[0-9]{1,3}[.)])\s+")

_STOPWORDS = set((
    "the","a","an","and","or","of","in","on","for","to","from","by","with","at","as","is",
    "are","was","were","be","been","it","its","this","that","these","those","we","you",
    "your","our","their","they","he","she","his","her","not","no","yes","but","if","then",
    "et","la","le","les","des","de","du","un","une","dans","pour","par","sur","au","aux",
))

def _extract_bullets(text: str) -> List[str]:
    """Return bullet-like lines from text."""
    out = []
    for line in (text or "").split("\n"):
        if _BULLET_RE.match(line):
            out.append(_BULLET_RE.sub("", line).strip())
    return out

def _top_keywords(text: str, k: int = 8) -> List[str]:
    """Super-light keyphrase heuristic: frequency of non-stopword >= 4 chars."""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9\-\_]{2,}", (text or "").lower())
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) < 4 or w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

# --------------------------- Qdrant-friendly IDs ---------------------------- #

# A fixed namespace for brochure chunks; keep constant to ensure UUIDv5 stability
_BROCHURE_NAMESPACE = uuid.UUID("33b7f3e3-5c3d-4f6a-9a1d-4a8e2d0f5d2c")

def _document_fingerprint(pages_ordered: List[List[Dict[str, Any]]], sample_pages: int = 2) -> uuid.UUID:
    """
    Build a deterministic UUID for the *document*, based on the first N pages' textual surface.
    This avoids needing the absolute file path.
    """
    texts: List[str] = []
    for elems in pages_ordered[:max(1, sample_pages)]:
        ordered = sorted(elems, key=lambda b: int(b.get("reading_order", 0)))
        for b in ordered:
            t = (b.get("text") or "").strip()
            if t:
                texts.append(t)
    sample = "\n".join(texts)[:4000]
    return uuid.uuid5(_BROCHURE_NAMESPACE, sample or "empty-doc")

def _page_chunk_uuid(doc_uuid: uuid.UUID, page_number: int) -> str:
    """Derive a page-level chunk UUID from the doc UUID; Qdrant-acceptable UUID string."""
    return str(uuid.uuid5(doc_uuid, f"page-{page_number:06d}"))

# ------------------------------ bbox helpers -------------------------------- #

def _union_bbox(elems: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Union of element bboxes; returns (x0,y0,x1,y1)."""
    x0 = min((float(b.get("bbox", [0,0,0,0])[0]) for b in elems), default=0.0)
    y0 = min((float(b.get("bbox", [0,0,0,0])[1]) for b in elems), default=0.0)
    x1 = max((float(b.get("bbox", [0,0,0,0])[2]) for b in elems), default=0.0)
    y1 = max((float(b.get("bbox", [0,0,0,0])[3]) for b in elems), default=0.0)
    return (x0, y0, x1, y1)

def _shrink_bbox(bbox: Tuple[float, float, float, float], ratio: float = 0.02) -> List[int]:
    """
    Inset bbox by a small ratio (2% default) on each side.
    Guarantees integer output and a valid box (at least 1px wide/high).
    """
    x0, y0, x1, y1 = bbox
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    dx = w * max(0.0, min(0.25, ratio))
    dy = h * max(0.0, min(0.25, ratio))
    nx0 = x0 + dx
    ny0 = y0 + dy
    nx1 = x1 - dx
    ny1 = y1 - dy
    # Ensure valid after shrink
    if nx1 <= nx0:
        midx = (x0 + x1) / 2.0
        nx0, nx1 = midx - 0.5, midx + 0.5
    if ny1 <= ny0:
        midy = (y0 + y1) / 2.0
        ny0, ny1 = midy - 0.5, midy + 0.5
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]

def _page_bbox(elems: List[Dict[str, Any]], shrink_ratio: float = 0.02) -> List[int]:
    """
    Page-level bbox approximated from element union, then slightly shrunk.
    This makes the bbox 'slightly smaller than the whole page' without needing page size.
    """
    if not elems:
        return [0, 0, 1, 1]
    union = _union_bbox(elems)
    return _shrink_bbox(union, ratio=shrink_ratio)

# ------------------------ page → brochure chunk logic ----------------------- #

def _collect_page_fields(elems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Split a page into headings, paragraphs, captions, images, and collect layout signals.
    Assumes `elems` belong to a single page.
    """
    ordered = sorted(elems, key=lambda b: int(b.get("reading_order", 0)))

    headings, paras, captions, images = [], [], [], []
    col_ids = set()

    for b in ordered:
        lbl = (b.get("label") or "").lower()
        txt = _normalize_lines(b.get("text") or "")
        if "col" in b and isinstance(b["col"], int) and b["col"] >= 0:
            col_ids.add(b["col"])

        if _is_image_label(lbl):
            fp = b.get("figure_path")
            images.append({
                "path": fp if fp else None,
                "bbox": [int(round(v)) for v in (b.get("bbox") or [0, 0, 0, 0])],
            })
        elif _is_caption_label(lbl):
            if txt:
                captions.append(txt)
        elif _is_heading_label(lbl):
            if txt:
                headings.append(txt)
        elif _is_paragraph_label(lbl) or lbl not in {"header", "footer"}:
            if txt:
                paras.append(txt)

    surface_pieces: List[str] = []
    if headings:
        surface_pieces.append(" / ".join(_squeeze(h) for h in headings))
    if paras:
        surface_pieces.append("\n\n".join(paras))
    if captions:
        surface_pieces.append("\n\nCAPTIONS:\n" + "\n".join(captions))
    if images:
        surface_pieces.append(f"\n\n[images:{len(images)}]")

    surface_text = _normalize_lines("\n\n".join(p for p in surface_pieces if p))

    bullets: List[str] = []
    for para in paras:
        bullets.extend(_extract_bullets(para))

    keywords = _top_keywords(surface_text, k=10)

    return {
        "headings": headings,
        "paragraphs": paras,
        "captions": captions,
        "images": images,
        "columns": len(col_ids) if col_ids else (1 if elems else 0),
        "surface_text": surface_text,
        "bullets": bullets,
        "keywords": keywords,
        "word_count": len(re.findall(r"\w+", surface_text)),
    }

# ------------------------------ public API --------------------------------- #

def build_chunks(pages_ordered: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Core entry point discovered by `classify_type_and_postprocess`.

    Output schema (per page):
      {
        "chunk_id": "<UUIDv5>",              # Qdrant point id
        "type": "brochure",
        "page_number": <int>,
        "doc_fingerprint": "<UUIDv5>",
        "bbox": [x0,y0,x1,y1],               # slightly smaller than page union
        "text": "<string>",                  # embedding surface
        "titles": [...],
        "captions": [...],
        "bullets": [...],
        "images": [{"path":..., "bbox":[...]}],
        "keywords": [...],
        "layout": {"columns": <int>, "reading_ordered": true},
        "stats": {"num_words": <int>, "num_images": <int>},
        "filters": {"doc_type":"brochure","page":<int>,"has_images":<bool>,"column_count":<int>}
      }
    """
    if not isinstance(pages_ordered, list):
        raise TypeError("build_chunks expects a list-of-pages (pages_ordered).")

    doc_uuid = _document_fingerprint(pages_ordered)
    chunks: List[Dict[str, Any]] = []

    for page_idx, elems in enumerate(pages_ordered, start=1):
        fields = _collect_page_fields(elems)
        chunk_id = _page_chunk_uuid(doc_uuid, page_idx)
        page_bbox = _page_bbox(elems, shrink_ratio=0.02)  # 2% inward inset

        chunk: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "type": "brochure",
            "page_number": page_idx,
            "doc_fingerprint": str(doc_uuid),
            "bbox": page_bbox,                 # <-- new attribute
            "text": fields["surface_text"],
            "titles": fields["headings"],
            "captions": fields["captions"],
            "bullets": fields["bullets"],
            "images": fields["images"],
            "keywords": fields["keywords"],
            # "layout": {
            #     "columns": int(fields["columns"]),
            #     "reading_ordered": True,
            # },
            "stats": {
                "num_words": int(fields["word_count"]),
                "num_images": int(len(fields["images"])),
            },
            "filters": {
                "doc_type": "brochure",
                "page": page_idx,
                "has_images": bool(fields["images"]),
                "column_count": int(fields["columns"]),
            },
        }

        chunks.append(chunk)

    return chunks

__all__ = ["build_chunks"]
