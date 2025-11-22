# cv_generate_chunks.py
# -----------------------------------------------------------------------------
# CV post-processor (one chunk per person):
# - Groups elements across pages into per-person profiles.
# - Stable UUIDv5 chunk IDs (doc_fingerprint + person slug).
# - Embedding-focused `text`: "name — primary expertise — N years. EN summary. [FR bref:] …"
# - Ensures renderer-required fields exist, but NEVER writes empty file meta:
#   If file meta env vars are missing, keys are omitted so upstream can fill them.
# - Callable via `build_chunks(pages_ordered)` and testable via CLI `main()`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import uuid
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------- small utilities ------------------------------ #

def _squeeze(s: str) -> str:
    return re.sub(r"[ \t\u00A0]+", " ", (s or "").strip())

def _normalize_lines(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(_squeeze(line) for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _sentences(text: str, max_sents: int | None = None) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", _squeeze(text))
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:max_sents] if max_sents else parts

_STOPWORDS = {
    "the","a","an","and","or","of","in","on","for","to","from","by","with","at","as","is","are","was","were","be","been","it","its",
    "this","that","these","those","we","you","your","our","their","they","he","she","his","her","not","no","yes","but","if","then",
    "et","la","le","les","des","de","du","un","une","dans","pour","par","sur","au","aux","en","delle","del","dei"
}

def _top_keywords(text: str, k: int = 10) -> List[str]:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9\-\_]{3,}", (text or "").lower())
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) < 4 or w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

def _looks_french(s: str) -> bool:
    s_low = (s or "").lower()
    if any(ch in s_low for ch in ("é","è","à","ê","ç","î","ô","û","ï","â")):
        return True
    fr_markers = ("ingénieur","barrage","étude","projet","centrale","années","conception","modélisation","risque","assainissement")
    return any(tok in s_low for tok in fr_markers)

def _ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    non_ascii = sum(1 for ch in s if ord(ch) > 127)
    return 1.0 - non_ascii / max(1, len(s))

def _slugify_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name.lower() or "unknown"

def _extract_name(text: str) -> str | None:
    t = _squeeze(text or "")
    t = re.split(r"\s+[–-]\s+[A-Z0-9]{2,6}$", t)[0]
    m = re.match(r"^([A-Z][A-Z\-\s']{1,}\s+[A-Z][a-zÀ-ÖØ-öø-ÿ][^,]*)", t)
    if m: return m.group(1).strip()
    m = re.match(r"^([A-Z][a-zÀ-ÖØ-öø-ÿ]+(?:\s+[A-Z][a-zÀ-ÖØ-öø-ÿ\-]+){0,3})$", t)
    if m: return m.group(1).strip()
    return None

def _extract_first_year(text: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    return int(m.group(0)) if m else None

def _extract_years_of_exp(text: str) -> int | None:
    m = re.search(r"experience[^0-9]*(\d{1,2})", (text or "").lower())
    return int(m.group(1)) if m else None

def _extract_primary_from_sentences(sents: List[str]) -> str | None:
    for s in sents:
        m = re.search(r"\b(speciali[sz]ed in)\s+([^.;:]+)", s, flags=re.I)
        if m: return _squeeze(m.group(2))
        m = re.search(r"\b(spécialisé(?:e)? en)\s+([^.;:]+)", s, flags=re.I)
        if m: return _squeeze(m.group(2))
    return None

def _shorten(s: str, max_chars: int) -> str:
    s = _squeeze(s or "")
    return s if len(s) <= max_chars else (s[: max_chars - 1].rstrip() + "…")

# --------------------------- bbox + size helpers ---------------------------- #

def _union_boxes(boxes: List[List[float]]) -> Tuple[float, float, float, float]:
    xs0 = [float(b[0]) for b in boxes if b and len(b) == 4]
    ys0 = [float(b[1]) for b in boxes if b and len(b) == 4]
    xs1 = [float(b[2]) for b in boxes if b and len(b) == 4]
    ys1 = [float(b[3]) for b in boxes if b and len(b) == 4]
    if not xs0 or not ys0 or not xs1 or not ys1:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))

def _shrink_bbox(bbox: Tuple[float, float, float, float], ratio: float = 0.02) -> List[int]:
    x0, y0, x1, y1 = bbox
    w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
    dx = w * max(0.0, min(0.25, ratio)); dy = h * max(0.0, min(0.25, ratio))
    nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
    if nx1 <= nx0:
        midx = (x0 + x1) / 2.0; nx0, nx1 = midx - 0.5, midx + 0.5
    if ny1 <= ny0:
        midy = (y0 + y1) / 2.0; ny0, ny1 = midy - 0.5, midy + 0.5
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]

def _page_size_from_elems(elems: List[Dict[str, Any]]) -> List[float]:
    if not elems:
        return [595.28, 841.89]
    boxes = [e.get("bbox") for e in elems if e.get("bbox")]
    _, _, x1, y1 = _union_boxes(boxes) if boxes else (0.0, 0.0, 595.28, 841.89)
    return [float(ceil(x1)), float(ceil(y1))]

# --------------------------- stable IDs / fingerprint ----------------------- #

_CV_NAMESPACE = uuid.UUID("b3240b25-3b7f-4d7e-98e1-5b7c6cdb7c1d")

def _document_fingerprint(pages_ordered: List[List[Dict[str, Any]]], sample_pages: int = 2) -> uuid.UUID:
    texts: List[str] = []
    for elems in pages_ordered[: max(1, sample_pages)]:
        ordered = sorted(elems, key=lambda b: int(b.get("reading_order", 0)))
        for b in ordered:
            t = (b.get("text") or "").strip()
            if t: texts.append(t)
    sample = "\n".join(texts)[:4000] or "empty-doc"
    return uuid.uuid5(_CV_NAMESPACE, sample)

def _person_chunk_uuid(doc_uuid: uuid.UUID, name_slug: str) -> str:
    return str(uuid.uuid5(doc_uuid, f"person:{name_slug}"))

# ------------------------------ main parsing -------------------------------- #

def _flatten(pages_ordered: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for pnum, elems in enumerate(pages_ordered, start=1):
        ordered = sorted(elems, key=lambda b: int(b.get("reading_order", 0)))
        for b in ordered:
            flat.append({
                "page": pnum,
                "label": (b.get("label") or "").lower(),
                "text": _normalize_lines(b.get("text") or ""),
                "bbox": b.get("bbox"),
            })
    return flat

def _guess_company(first_page_elems: List[Dict[str, Any]]) -> str | None:
    for b in first_page_elems[:8]:
        t = (b.get("text") or "").strip()
        if re.search(r"\b(AG|SA|GmbH|Ltd\.?|S\.?A\.?|SARL|Sàrl)\b", t):
            return _squeeze(t)
    return None

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

# ------------------------------ public API --------------------------------- #

def build_chunks(pages_ordered: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Input: list-of-pages (each page is list of blocks with 'label','text','reading_order','bbox').
    Output: one chunk per person; never writes empty file meta (lets upstream fill them).
    """
    if not isinstance(pages_ordered, list):
        raise TypeError("build_chunks expects a list-of-pages (pages_ordered).")

    doc_uuid = _document_fingerprint(pages_ordered)
    flat = _flatten(pages_ordered)
    company = _guess_company(pages_ordered[0]) if pages_ordered else None
    file_meta_env = _file_meta_from_env()
    file_meta_present = {k: v for k, v in file_meta_env.items() if v}  # keep only non-empty

    profiles: List[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None

    def _ensure_profile(name_text: str, page: int):
        nonlocal cur
        name = _extract_name(name_text) or name_text.strip()
        if cur and _slugify_name(cur["name"]) == _slugify_name(name):
            cur.setdefault("aliases", set()).add(name_text.strip())
            cur["page_end"] = max(cur["page_end"], page)
            return
        if cur and cur.get("name"):
            profiles.append(cur)
        cur = {
            "name": name,
            "aliases": set([name_text.strip()]),
            "page_start": page,
            "page_end": page,
            "first_diploma_year": None,
            "years_of_experience": None,
            "edu_lines": [],
            "sum_en": [],
            "sum_fr": [],
            "raw_text": [],
            "roles": set(),
            "geos": set(),
            "page_boxes": {},
        }

    for b in flat:
        t = b["text"]; lbl = b["label"]; page = b["page"]
        is_name_heading = (lbl == "sub_sub_sec" and _extract_name(t)) or (
            lbl in {"para", "sec"} and re.search(r"\s[–-]\s*[A-Z]{2,6}$", t) and _extract_name(t)
        ) or (lbl == "para" and bool(re.match(r"^[A-Z][A-Z\-\s']+\s+[A-Z][a-z]", t)))
        if is_name_heading:
            _ensure_profile(t, page)
            continue

        if not cur:
            continue

        cur["raw_text"].append(t)
        cur["page_end"] = max(cur["page_end"], page)
        if b.get("bbox"):
            cur["page_boxes"].setdefault(page, []).append(b["bbox"])

        if "year of the first diploma" in t.lower():
            y = _extract_first_year(t)
            if y: cur["first_diploma_year"] = y
            continue

        if "years of experience" in t.lower():
            yx = _extract_years_of_exp(t)
            if yx is not None: cur["years_of_experience"] = yx
            continue

        if t.lower().startswith("education:") or t.lower().startswith("education :"):
            for line in (t.split("\n")[1:] or []):
                line = _squeeze(line)
                if line: cur["edu_lines"].append(line)
            continue

        if _looks_french(t):
            cur["sum_fr"].append(t)
        else:
            cur["sum_en"].append(t)

        for g in ("Switzerland","France","Senegal","Burkina Faso","Ecuador","Tajikistan","Lyon","Dakar","South America"):
            if g.lower() in t.lower(): cur["geos"].add(g)

        for r in ("consultant","contractor designer","project manager","branch coordinator","engineer","specialist"):
            if r in t.lower(): cur["roles"].add(r)

    if cur and cur.get("name"):
        profiles.append(cur)

    chunks: List[Dict[str, Any]] = []
    for p in profiles:
        name = p["name"]
        name_slug = _slugify_name(name)
        chunk_id = _person_chunk_uuid(doc_uuid, name_slug)

        # Education
        education = []
        for line in p["edu_lines"]:
            m = re.match(r"^([A-Za-z\.]+)\s+(?:in|en)?\s*(.*?)(?:\s*\((\d{4})\))?$", line)
            if m:
                degree = (m.group(1) or "").strip().rstrip(".") or None
                field = _squeeze(m.group(2) or "") or None
                year = int(m.group(3)) if m.group(3) else None
                education.append({"degree": degree, "field": field, "year": year})
            else:
                y = _extract_first_year(line)
                education.append({"degree": None, "field": _squeeze(line), "year": y})

        en_sents = _sentences(" ".join(p["sum_en"]))
        fr_sents = _sentences(" ".join(p["sum_fr"]))
        primary = _extract_primary_from_sentences(en_sents or fr_sents)
        if not primary:
            kws = _top_keywords(" ".join(p["sum_en"] + p["sum_fr"]))[:3]
            primary = " / ".join(kws) if kws else "general engineering"

        skills = _top_keywords(" ".join(p["raw_text"]))[:8]
        en_summary = _shorten(" ".join(_sentences(" ".join(p["sum_en"]) or " ".join(p["raw_text"]), max_sents=3)), 600)
        fr_brief = _shorten(fr_sents[0], 220) if fr_sents else ""
        langs = (["EN"] if p["sum_en"] else []) + (["FR"] if p["sum_fr"] else [])
        years_exp = p.get("years_of_experience")
        years_str = f"{years_exp} years" if isinstance(years_exp, int) else "n/a years"

        # Renderer-required spatials
        start_page = int(p["page_start"])
        prof_boxes = p.get("page_boxes", {}).get(start_page, [])
        if prof_boxes:
            page_bbox = _shrink_bbox(_union_boxes(prof_boxes), ratio=0.02)
        else:
            all_boxes = [e.get("bbox") for e in (pages_ordered[start_page - 1] if 0 < start_page <= len(pages_ordered) else []) if e.get("bbox")]
            page_bbox = _shrink_bbox(_union_boxes(all_boxes), ratio=0.02) if all_boxes else [0, 0, 1, 1]
        orig_size = _page_size_from_elems(pages_ordered[start_page - 1] if 0 < start_page <= len(pages_ordered) else [])

        text_surface = f"{name} — {primary} — {years_str}. {en_summary}"
        if fr_brief:
            text_surface += f" [FR bref:] {fr_brief}"

        metadata = {
            "doc_fingerprint": str(doc_uuid),
            "page_span": [int(p["page_start"]), int(p["page_end"])],
            "name": name,
            "aliases": sorted(a for a in p.get("aliases", set()) if a and a != name),
            "company": company,
            "first_diploma_year": p.get("first_diploma_year"),
            "years_of_experience": years_exp,
            "primary_expertise": [seg.strip() for seg in re.split(r"[,/;]| and ", primary) if seg.strip()][:5],
            "skills": skills,
            "education": education,
            "languages_present": langs or ["EN"],
            "geographies": sorted(p.get("geos", set())),
            "roles": sorted(p.get("roles", set())),
            "section": "Key Qualifications",
            "entity_type": "cv_profile",
        }

        chunk: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "type": "cv_profile",
            "text": text_surface,
            "metadata": metadata,
            "page_number": start_page,
            "doc_fingerprint": str(doc_uuid),
            "bbox": page_bbox,
            "orig_size": orig_size,
        }

        # >>> IMPORTANT: only attach file meta if non-empty to avoid blocking upstream defaults
        # (create_chunks_any will .setdefault('file_id','file_name') and will derive source_* from them)
        for k, v in file_meta_present.items():
            chunk[k] = v  # k in {"file_id","file_name","source_file_id","source_file_name","content_hash"}

        chunks.append(chunk)

    return chunks

# ------------------------------ CLI / testing ------------------------------- #

def _as_pages_ordered(data: Any) -> List[List[Dict[str, Any]]]:
    if isinstance(data, dict) and "pages" in data:
        pages = sorted(data.get("pages") or [], key=lambda p: p.get("page_number", 0))
        return [p.get("elements") or [] for p in pages]
    if isinstance(data, list) and data:
        if isinstance(data[0], list): return data
        if isinstance(data[0], dict) and "elements" in data[0]:
            pages = sorted(data, key=lambda p: p.get("page_number", 0))
            return [p.get("elements") or [] for p in pages]
    if isinstance(data, list) and not data:
        return []
    raise TypeError("Unsupported input JSON shape. Expect dict with 'pages', or list of pages/elements.")

def _sample_doc() -> Dict[str, Any]:
    return {
        "pages": [
            {
                "page_number": 1,
                "elements": [
                    {"label":"para","bbox":[71,80,161,91],"text":"Gruner Stucky Ltd","reading_order":0},
                    {"label":"para","bbox":[78,105,209,116],"text":"ABATI Andrea (Dr) – ABAN","reading_order":1},
                    {"label":"sec","bbox":[73,125,418,139],"text":"Year of the first diploma / Année du premier diplôme :\n2004","reading_order":2},
                    {"label":"sec","bbox":[73,143,423,161],"text":"Years of experience / Années d'expérience :\n21","reading_order":3},
                    {"label":"para","bbox":[76,170,445,225],"text":"Education:\nPhD in Structural and Geotechnical Engineering (2008)\nM.Sc. in Civil Engineering (2004)","reading_order":4},
                    {"label":"para","bbox":[197,231,791,409],"text":"Dr Andrea Abati is a civil engineer specialized in the design of dams and appurtenant structures, from prefeasibility to construction and rehabilitation, with expertise spanning structural and geotechnical aspects, tunnels and penstocks.","reading_order":5},
                    {"label":"para","bbox":[76,431,451,486],"text":"Ingénieur civil spécialisé en barrages et ouvrages annexes, analyses numériques et aspects de construction.","reading_order":6},
                ],
            }
        ]
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Build CV chunks (one per person) from parsed pages JSON.")
    parser.add_argument("-i", "--input", type=str, help="Input JSON path (doc with 'pages' or list). If omitted, uses a tiny sample.")
    parser.add_argument("-o", "--output", type=str, default="cv_chunks.json", help="Output JSON file (default: cv_chunks.json)")
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")
        data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    else:
        data = _sample_doc()

    pages_ordered = _as_pages_ordered(data)
    chunks = build_chunks(pages_ordered)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunk(s) → {out_path}")

if __name__ == "__main__":
    main()

__all__ = ["build_chunks", "main"]
