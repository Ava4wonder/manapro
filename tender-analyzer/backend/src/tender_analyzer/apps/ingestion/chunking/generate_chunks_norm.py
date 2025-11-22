#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
norm_generate_chunks.py
-----------------------
Refactored DE-OCF chunker as a module exposing `build_chunks(pages_ordered, *, source_file=None, lang="fr")`
so it can be called directly by `classify_type_and_postprocess` for the "norm" case.

- Keeps the original clause-level chunk schema and logic.
- Ensures each chunk contains `page_span` (with page number) and `bbox_span`.
- Can also run as a CLI compatible with the previous script (reads the old JSON format).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set

# --------------- Regex patterns ---------------

RE_AD_ART = re.compile(r'Ad\s*art\.?:\s*(\d+)', flags=re.IGNORECASE)
RE_DE_CODE = re.compile(r'\(?\bDE\s+([0-9]+(?:\.[0-9]+)*?(?:bis|ter|quater)?|\d+)\)?', flags=re.IGNORECASE)
RE_EDITION = re.compile(r'Edition:\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})', flags=re.IGNORECASE)
RE_CHAPITRE = re.compile(r'Chapitre:\s*(.+)', flags=re.IGNORECASE | re.DOTALL)
RE_ARTICLE_TITLE = re.compile(r'Article:\s*(.+)', flags=re.IGNORECASE | re.DOTALL)

RE_CLAUSE_NUMBER_LINE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s*$', flags=re.MULTILINE)
RE_LETTERED_LINE = re.compile(r'^\s*([a-k])\.\s+(.+)$', flags=re.IGNORECASE | re.DOTALL)

RE_RS_CODE = re.compile(r'RS\s+\d+(?:\.\d+)+')
RE_ISO = re.compile(r'\bISO\s+\d{3,5}\b', flags=re.IGNORECASE)
RE_SN_SIA = re.compile(r'\b(SN\s+\d+(?:\s+\d+)*|SIA\s+\d+)\b', flags=re.IGNORECASE)
RE_STI = re.compile(r'\bSTI\b', flags=re.IGNORECASE)
RE_ICT = re.compile(r'\bITU(?:-T)?\b', flags=re.IGNORECASE)

RE_DB_A = re.compile(r'(\d{2,3})\s*dB\s*\(\s*A\s*\)', flags=re.IGNORECASE)
RE_SPEED_KMH = re.compile(r'(\d{1,3})\s*km/h', flags=re.IGNORECASE)

RE_STATE_DATE_FR = re.compile(r"Etat au\s+(.+)", flags=re.IGNORECASE)

# --------------- Utilities ---------------

def normalize_whitespace(s: str) -> str:
    s = s.replace('\r', '\n')
    s = re.sub(r'\n{2,}', '\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def strip_page_furniture(text: str) -> str:
    text = re.sub(r'^DISPOSITIONS.*CHEMINS DE FER.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Feuille n°:\s*\d+\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Edition:\s*\d{2}\.\d{2}\.\d{4}\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Chapitre:.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Article:.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Ad art\.:.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return normalize_whitespace(text)

FR_MONTHS = {
    'janvier': '01', 'février': '02', 'fevrier': '02', 'mars': '03', 'avril': '04',
    'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08', 'aout': '08',
    'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12', 'decembre': '12'
}

def parse_date_iso(s: str) -> Optional[str]:
    s = s.strip()
    m = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', s)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"
    s = s.lower().replace('1er', '1')
    m = re.match(r'(\d{1,2})\s+([a-zéèêûôîàùç]+)\s+(\d{4})', s)
    if m:
        d, month_name, yyyy = m.groups()
        mm = FR_MONTHS.get(month_name)
        if mm:
            return f"{yyyy}-{mm}-{int(d):02d}"
    return None

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s

def split_long_text(text: str, max_chars: int = 1600, overlap_chars: int = 120) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    sentences = re.split(r'(?<=[\.\!\?\:])\s+', text)
    cur = ""
    for sent in sentences:
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur += " " + sent
        else:
            parts.append(cur.strip())
            overlap = cur[-overlap_chars:]
            cur = overlap + " " + sent
            if len(cur) > max_chars:
                parts.append(cur[:max_chars])
                cur = cur[max_chars-overlap_chars:]
    if cur.strip():
        parts.append(cur.strip())
    parts = [p[:max_chars] for p in parts]
    return parts

def extract_refs_and_norms(text: str) -> Tuple[Set[str], Set[str]]:
    refs = set(RE_RS_CODE.findall(text))
    norms = set(RE_ISO.findall(text))
    norms.update([m.group(0) for m in RE_SN_SIA.finditer(text)])
    if RE_STI.search(text):
        norms.add("STI")
    if RE_ICT.search(text):
        norms.add("ITU")
    return refs, norms

def extract_numbers(text: str) -> Dict[str, Any]:
    nums: Dict[str, Any] = {}
    dbs = [int(x) for x in RE_DB_A.findall(text)]
    if dbs:
        nums["noise_dBA"] = dbs
    speeds = [int(x) for x in RE_SPEED_KMH.findall(text)]
    if speeds:
        nums["speeds_kmh"] = speeds
    return nums

# --------------- Helpers over pages_ordered ---------------

def element_sort_key(el: Dict[str, Any]) -> Tuple[int, int]:
    return (int(el.get("reading_order", 0)), 0)

def page_elements_sorted_from_ordered(page_elems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    els = [e for e in page_elems if isinstance(e.get("text", ""), str)]
    return sorted(els, key=lambda e: int(e.get("reading_order", 0)))

def detect_context_from_page(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"ad_art": None, "edition": None, "chapter": None, "article_title": None}
    joined = "\n".join(e.get("text", "") for e in elements if isinstance(e.get("text"), str))
    m = RE_AD_ART.search(joined)
    if m:
        ctx["ad_art"] = int(m.group(1))
    m = RE_EDITION.search(joined)
    if m:
        ctx["edition"] = parse_date_iso(m.group(1))
    m = RE_CHAPITRE.search(joined)
    if m:
        ctx["chapter"] = normalize_whitespace(m.group(1))
    m = RE_ARTICLE_TITLE.search(joined)
    if m:
        ctx["article_title"] = normalize_whitespace(m.group(1))
    return ctx

def detect_state_date_from_ordered(pages_ordered: List[List[Dict[str, Any]]]) -> Optional[str]:
    for page in pages_ordered:
        for el in page:
            t = el.get("text", "")
            if not isinstance(t, str):
                continue
            m = RE_STATE_DATE_FR.search(t)
            if m:
                iso = parse_date_iso(m.group(1))
                if iso:
                    return iso
    return None

def find_doc_title_from_ordered(pages_ordered: List[List[Dict[str, Any]]]) -> str:
    candidates: List[str] = []
    for page in pages_ordered:
        for el in page:
            t = el.get("text", "")
            if not isinstance(t, str):
                continue
            if ("Dispositions d'exécution" in t) or ("DISPOSITIONS D’EXÉCUTION" in t) or ("ORDONNANCE" in t):
                candidates.append(normalize_whitespace(t))
    if candidates:
        return max(candidates, key=len)[:200]
    return "DE-OCF – Dispositions d’exécution de l’OCF"

def is_de_marker(el_text: str) -> Optional[str]:
    m = RE_DE_CODE.search(el_text)
    return m.group(1) if m else None

def clause_from_para_text(text: str) -> Tuple[Optional[str], str]:
    text = normalize_whitespace(text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None, ""
    if re.match(r'^\d+(?:\.\d+)*$', lines[0]):
        clause = lines[0]
        body = normalize_whitespace("\n".join(lines[1:]))
        return clause, body
    return None, text

def lettered_item_from_text(text: str) -> Optional[Tuple[str, str]]:
    m = RE_LETTERED_LINE.match(text.strip())
    if m:
        return m.group(1).lower(), normalize_whitespace(m.group(2))
    return None

def build_chunk_id(prefix: str, article_ocf: Optional[int], de_code: Optional[str],
                   clause: Optional[str], list_item: Optional[str], part: Optional[int] = None) -> str:
    bits: List[str] = [prefix]
    if article_ocf is not None:
        bits.append(f"art{article_ocf}")
    if de_code:
        bits.append(f"de{de_code.replace(' ', '').lower()}")
    if clause:
        bits.append(clause.replace('.', '_'))
    if list_item:
        bits.append(list_item)
    if part is not None:
        bits.append(f"part{part}")
    return ".".join(bits)

def build_breadcrumbs(chapter: Optional[str], article_ocf: Optional[int],
                      de_code: Optional[str], clause: Optional[str]) -> List[str]:
    crumbs: List[str] = []
    if chapter:
        crumbs.append(chapter)
    if article_ocf is not None:
        crumbs.append(f"Ad art. {article_ocf}")
    if de_code:
        crumbs.append(f"DE {de_code}")
    if clause:
        crumbs.append(f"Clause {clause}")
    return crumbs

def classify_tags(text: str) -> List[str]:
    t = text.lower()
    tags = set()
    for kw, tag in [
        ("cybersecur", "cybersécurité"),
        ("smsi", "SMSI"),
        ("sgs", "SGS"),
        ("bruit", "bruit"),
        ("emission", "émissions"),
        ("tram", "tram"),
        ("locomotive", "locomotives"),
        ("uic", "UIC"),
        ("interoperab", "interopérabilité"),
        ("iso", "ISO"),
    ]:
        if kw in t:
            tags.add(tag)
    return sorted(tags)

def make_chunk(record_template: Dict[str, Any], text: str,
               clause: Optional[str], list_item: Optional[str],
               article_ocf: Optional[int], de_code: Optional[str],
               page_no: int, bbox: Optional[List[int]],
               page_refs: Iterable[str], page_norms: Iterable[str]) -> List[Dict[str, Any]]:
    clean = strip_page_furniture(text)
    if not clean:
        return []
    refs1, norms1 = extract_refs_and_norms(clean)
    nums = extract_numbers(clean)
    chunks: List[Dict[str, Any]] = []
    parts = split_long_text(clean, max_chars=1600, overlap_chars=120)
    for idx, part_text in enumerate(parts):
        rec = {
            **record_template,
            "id": build_chunk_id("de-ocf", article_ocf, de_code, clause, list_item, idx if len(parts) > 1 else None),
            "breadcrumbs": build_breadcrumbs(record_template.get("breadcrumbs_base_chapter"), article_ocf, de_code, clause),
            "article_ocf": article_ocf,
            "de_code": de_code,
            "clause_number": clause,
            "list_item": list_item,
            "page_span": [page_no, page_no],
            "bbox_span": bbox,
            "text": part_text,
            "refs": sorted(set(record_template.get("refs_base", [])) | set(page_refs) | refs1),
            "norms": sorted(set(record_template.get("norms_base", [])) | set(page_norms) | norms1),
            "numbers": nums if nums else {},
            "tags": classify_tags(part_text),
        }
        chunks.append(rec)
    return chunks

# --------------- Public API: build_chunks over pages_ordered ---------------

def build_chunks(pages_ordered: List[List[Dict[str, Any]]],
                 *, source_file: Optional[str] = None, lang: str = "fr") -> List[Dict[str, Any]]:
    """
    Build DE-OCF clause-level chunks from `pages_ordered` (list of pages; each page is a list of blocks).
    This is designed to be called by `classify_type_and_postprocess` for the "norm" path.

    Returns: List[chunk dicts] with the original schema:
      id, doc_title, source_file, breadcrumbs[], article_ocf, de_code, clause_number, list_item,
      edition_date, state_date, page_span, bbox_span, text, refs[], norms[], numbers{}, lang, tags[]
    """
    if not isinstance(pages_ordered, list):
        raise TypeError("build_chunks expects a list-of-pages (pages_ordered).")

    doc_title = find_doc_title_from_ordered(pages_ordered)
    state_date = detect_state_date_from_ordered(pages_ordered)

    all_chunks: List[Dict[str, Any]] = []
    current_ad_art: Optional[int] = None
    current_edition: Optional[str] = None
    current_chapter: Optional[str] = None
    current_article_title: Optional[str] = None
    current_de_code: Optional[str] = None

    base_template = {
        "doc_title": doc_title[:200],
        "source_file": source_file,
        "state_date": state_date,
        "lang": lang,
        "breadcrumbs_base_chapter": None,
        "refs_base": [],
        "norms_base": [],
    }

    for page_idx, page_elems in enumerate(pages_ordered, start=1):
        els = page_elements_sorted_from_ordered(page_elems)
        ctx = detect_context_from_page(els)

        current_ad_art = ctx.get("ad_art") or current_ad_art
        current_edition = ctx.get("edition") or current_edition
        current_chapter = ctx.get("chapter") or current_chapter
        current_article_title = ctx.get("article_title") or current_article_title

        record_template = {**base_template}
        record_template["edition_date"] = current_edition
        record_template["breadcrumbs_base_chapter"] = current_chapter

        # Collect footer refs/norms for this page
        page_refs: Set[str] = set()
        page_norms: Set[str] = set()
        for el in els:
            if el.get("label") == "footer":
                r, n = extract_refs_and_norms(el.get("text", ""))
                page_refs |= r
                page_norms |= n

        last_clause_seen: Optional[str] = None

        for el in els:
            t = el.get("text", "")
            if not isinstance(t, str):
                continue

            # Update DE code if marker present
            de = is_de_marker(t)
            if de:
                current_de_code = de

            # Skip pure furniture (keep logic from original)
            if el.get("label") in ("sub_sec",) and ("ARTICLE OCF" in t or "SUIVI" in t or "ANNEXES" in t):
                continue

            # Lettered list item following a clause
            li = lettered_item_from_text(t)
            if li and (last_clause_seen is not None):
                letter, body = li
                chunks = make_chunk(
                    {
                        **record_template,
                        "breadcrumbs": [],
                    },
                    body,
                    last_clause_seen,
                    letter,
                    current_ad_art,
                    current_de_code,
                    page_idx,
                    el.get("bbox"),
                    page_refs,
                    page_norms,
                )
                all_chunks.extend(chunks)
                continue

            # Clause-numbered paragraph
            clause_num, body = clause_from_para_text(t)
            if clause_num:
                last_clause_seen = clause_num
                chunks = make_chunk(
                    {
                        **record_template,
                        "breadcrumbs": [],
                    },
                    body,
                    clause_num,
                    None,
                    current_ad_art,
                    current_de_code,
                    page_idx,
                    el.get("bbox"),
                    page_refs,
                    page_norms,
                )
                all_chunks.extend(chunks)
                continue

            # Standalone informative paragraph within a DE section
            if current_de_code and t.strip():
                content = strip_page_furniture(t)
                if content:
                    note_clause = last_clause_seen
                    chunks = make_chunk(
                        {
                            **record_template,
                            "breadcrumbs": [],
                        },
                        content,
                        note_clause,
                        None,
                        current_ad_art,
                        current_de_code,
                        page_idx,
                        el.get("bbox"),
                        page_refs,
                        page_norms,
                    )
                    all_chunks.extend(chunks)

    # Cleanup helper fields to match schema
    for c in all_chunks:
        c.pop("breadcrumbs_base_chapter", None)
        c.pop("refs_base", None)
        c.pop("norms_base", None)

    return all_chunks

# Optional aliases so other norm hooks can find us
generate_norm_chunks = build_chunks
postprocess_norm = build_chunks
norm_postprocess = build_chunks

__all__ = ["build_chunks", "generate_norm_chunks", "postprocess_norm", "norm_postprocess"]

# --------------- Backward-compatible CLI ---------------

def _build_chunks_from_legacy_json(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Support running this module as a drop-in CLI, consuming the old JSON."""
    pages = doc.get("pages", [])
    # Transform legacy doc['pages'][i]['elements'] -> pages_ordered (sorted)
    po: List[List[Dict[str, Any]]] = []
    for p in pages:
        elems = p.get("elements", []) or []
        elems_sorted = sorted(
            [e for e in elems if isinstance(e.get("text", ""), str)],
            key=lambda e: int(e.get("reading_order", 0))
        )
        po.append(elems_sorted)
    return build_chunks(po, source_file=doc.get("source_file"), lang="fr")

def main():
    ap = argparse.ArgumentParser(description="Chunk DE-OCF parsed JSON (legacy) into clause-level chunks (JSONL)")
    ap.add_argument("input", help="path to input JSON (legacy format)")
    ap.add_argument("--out", default="chunks.jsonl", help="output JSONL path")
    ap.add_argument("--pretty", action="store_true", help="also write a pretty JSON summary")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        doc = json.load(f)

    chunks = _build_chunks_from_legacy_json(doc)

    with open(args.out, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    if args.pretty:
        from collections import defaultdict
        key = lambda x: (x.get("article_ocf"), x.get("de_code"))
        groups = defaultdict(list)
        for ch in chunks:
            groups[key(ch)].append(ch)
        pretty = {}
        for (art, de), items in groups.items():
            pretty_key = f"art{art}.de{de}"
            pretty[pretty_key] = [{
                "id": it["id"],
                "clause": it.get("clause_number"),
                "list_item": it.get("list_item"),
                "page": it.get("page_span"),
                "text_preview": it.get("text", "")[:140] + ("…" if len(it.get("text",""))>140 else "")
            } for it in items]
        pretty_path = args.out.rsplit(".",1)[0] + ".pretty.json"
        with open(pretty_path, "w", encoding="utf-8") as f:
            json.dump(pretty, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(chunks)} chunks to {args.out}")

if __name__ == "__main__":
    main()
