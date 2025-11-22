# coarse-to-fine
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, pathlib, argparse, math, importlib
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Any, Tuple
from dataclasses import dataclass

import subprocess, shutil
from pathlib import Path
import fitz

import os, sys, win32com.client
SAVE_IMAGE = False

@dataclass
class ParsedCell:
    row: int
    col: int
    bbox: Tuple[float, float, float, float]
    text: str
    kvs: List[Tuple[str, str]]
    bullets: List[str]
    selected_checkboxes: List[str]

def _parse_cell(page, cell_rect: fitz.Rect, row: int, col: int) -> ParsedCell:
    lines = _lines_from_clip(page, cell_rect)
    text = "\n".join(_norm(L["text"]) for L in lines if _norm(L["text"]))

    # 1) inline KVs (can be multiple in one cell)
    kvs = _extract_inline_kvs(lines)

    # 2) bullets
    bullets = _extract_bullets(lines)

    # 3) checkboxes (textual → drawing fallback)
    checked, _ = _extract_checkboxes_textual(lines)
    if not checked:
        # try drawing-based within this cell
        checked = _extract_checkboxes_drawn(page, cell_rect)

    return ParsedCell(
        row=row, col=col, bbox=(cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1),
        text=text, kvs=kvs, bullets=bullets, selected_checkboxes=checked
    )

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*(•|-)\s+", line))

def _kv_from_line(line: str) -> Optional[Tuple[str, str]]:
    if ":" in line:
        k, v = line.split(":", 1)
        k, v = _norm(k), _norm(v)
        if k and v:
            return (k, v)
    return None

def _boldish(span: Dict[str, Any]) -> bool:
    """Heuristic: 'Bold' (or 'Bd') in font name implies bold."""
    fname = span.get("font", "") or ""
    return ("Bold" in fname) or fname.endswith("-Bd") or "-Bold" in fname

def _lines_from_clip(page, rect) -> List[Dict[str, Any]]:
    """Return text lines from a clipped rectangle with span details."""
    d = page.get_text("dict", clip=rect)
    lines = []
    for block in d.get("blocks", []):
        for l in block.get("lines", []):
            spans = l.get("spans", [])
            txt = "".join(s.get("text", "") for s in spans)
            lines.append({"text": txt, "spans": spans})
    return lines

def pptx_to_pdf_win(input_path: str, output_path: str | None = None):
    input_path = os.path.abspath(input_path)
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".pdf"
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    powerpoint.Visible = 1
    try:
        pres = powerpoint.Presentations.Open(input_path, WithWindow=False)
        # 32 = PDF (per Office SaveAs constants)
        pres.SaveAs(output_path, 32)
        pres.Close()
    finally:
        powerpoint.Quit()
    return output_path

import platform, shutil
def pptx_to_pdf(input_path: str, output_path: str | None = None):
    if platform.system() == "Windows":
        return pptx_to_pdf_win(input_path, output_path)
    raise RuntimeError("No converter available: install LibreOffice or run on Windows with PowerPoint.")


def office_to_pdf(input_path: Path) -> Path:
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return input_path
    if ext == ".docx":
        return docx_to_pdf(input_path)   # your existing helper
    if ext == ".pptx":
        return pptx_to_pdf(input_path)
    raise ValueError(f"Unsupported input: {ext}")


# --- 1) DOCX -> PDF converters ------------------------------------------------
def _docx_to_pdf_word(docx_path: str, pdf_path: str):
    import win32com.client  # requires Microsoft Word installed
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path), ReadOnly=True)
        # 17 == wdFormatPDF
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
        doc.Close(False)
    finally:
        word.Quit()

def _docx_to_pdf_libreoffice(docx_path: str, out_dir: str):
    cmd = f'soffice --headless --convert-to pdf --outdir {shlex.quote(out_dir)} {shlex.quote(docx_path)}'
    subprocess.run(cmd, check=True, shell=True)
    base = os.path.splitext(os.path.basename(docx_path))[0]
    return os.path.join(out_dir, base + ".pdf")

def docx_to_pdf(docx_path: str, pdf_path: str | None = None) -> str:
    docx_path = os.path.abspath(docx_path)
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(docx_path)
    if pdf_path is None:
        base = os.path.splitext(docx_path)[0]
        pdf_path = base + ".pdf"

    system = platform.system()
    # Prefer Word on Windows; fallback to LibreOffice if available
    if system == "Windows":
        try:
            _docx_to_pdf_word(docx_path, pdf_path)
            return pdf_path
        except Exception as e:
            if shutil.which("soffice"):
                return _docx_to_pdf_libreoffice(docx_path, os.path.dirname(pdf_path))
            raise RuntimeError(f"Word/LibreOffice conversion failed: {e}") from e
    else:
        if shutil.which("soffice"):
            return _docx_to_pdf_libreoffice(docx_path, os.path.dirname(pdf_path))
        raise RuntimeError("No converter available. Install LibreOffice or run on Windows with Word.")
    
    
def _parse_tables_on_page(page) -> list[dict[str, any]]:
    def _find_tables(page, clip=None):
        try:
            return page.find_tables(clip=clip).tables
        except TypeError:
            return page.find_tables().tables

    tables_out: list[dict[str, any]] = []
    top_tables = _find_tables(page)

    for t in top_tables:
        t_bbox = list(fitz.Rect(t.bbox))
        cells_out: list[dict[str, any]] = []

        # ✅ PyMuPDF 1.26 API: iterate rows -> row.cells (each cell is a bbox or None)
        for r, row in enumerate(t.rows):
            for c, cbbox in enumerate(row.cells):
                if cbbox is None:
                    continue
                rect = fitz.Rect(cbbox)

                # detect nested sub-tables inside this cell (macro box case)
                nested = _find_tables(page, clip=rect)
                if nested:
                    for nt in nested:
                        for nr, nrow in enumerate(nt.rows):
                            for nc, ncbbox in enumerate(nrow.cells):
                                if ncbbox is None:
                                    continue
                                nrect = fitz.Rect(ncbbox)
                                parsed = _parse_cell(page, nrect, r, c)
                                cells_out.append({
                                    "row": r, "col": c, "bbox": list(nrect),
                                    "text": parsed.text,
                                    "kvs": parsed.kvs,
                                    "bullets": parsed.bullets,
                                    "selected_checkboxes": parsed.selected_checkboxes
                                })
                else:
                    parsed = _parse_cell(page, rect, r, c)
                    cells_out.append({
                        "row": r, "col": c, "bbox": list(rect),
                        "text": parsed.text,
                        "kvs": parsed.kvs,
                        "bullets": parsed.bullets,
                        "selected_checkboxes": parsed.selected_checkboxes
                    })

        tables_out.append({
            "bbox": t_bbox,
            "rows": t.row_count,
            "cols": t.col_count,
            "cells": cells_out
        })
    return tables_out

def _parse_cell(page, cell_rect: fitz.Rect, row: int, col: int) -> ParsedCell:
    lines = _lines_from_clip(page, cell_rect)
    text = "\n".join(_norm(L["text"]) for L in lines if _norm(L["text"]))

    # 1) inline KVs (can be multiple in one cell)
    kvs = _extract_inline_kvs(lines)

    # 2) bullets
    bullets = _extract_bullets(lines)

    # 3) checkboxes (textual → drawing fallback)
    checked, _ = _extract_checkboxes_textual(lines)
    if not checked:
        # try drawing-based within this cell
        checked = _extract_checkboxes_drawn(page, cell_rect)

    return ParsedCell(
        row=row, col=col, bbox=(cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1),
        text=text, kvs=kvs, bullets=bullets, selected_checkboxes=checked
    )

def _extract_inline_kvs(lines: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    kvs: List[Tuple[str, str]] = []
    for L in lines:
        t = _norm(L["text"])
        if not t:
            continue
        # 1) colon-based
        kv = _kv_from_line(t)
        if kv:
            kvs.append(kv)
            continue
        # 2) formatting-based: contiguous boldish then plain
        key_buf, val_buf, seen_plain = [], [], False
        for sp in L["spans"]:
            txt = sp.get("text", "")
            if not seen_plain and (_boldish(sp) or txt.strip().endswith(":")):
                key_buf.append(txt)
            else:
                seen_plain = True
                val_buf.append(txt)
        k, v = _norm("".join(key_buf)).rstrip(":"), _norm("".join(val_buf))
        if k and v:
            kvs.append((k, v))
    return kvs

def _extract_bullets(lines: List[Dict[str, Any]]) -> List[str]:
    items: List[str] = []
    for L in lines:
        t = _norm(L["text"])
        if _is_bullet(t):
            items.append(re.sub(r"^\s*(•|-)\s+", "", t))
    return items

CB_PAT = re.compile(r"^\s*(?:☑|☒|\[x\]|\[X\]|x\s|X\s|✓)\s*(.+)$")

def _extract_checkboxes_textual(lines: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Return (checked, unchecked) labels using text-only clues."""
    checked, unchecked = [], []
    for L in lines:
        t = _norm(L["text"])
        if not t:
            continue
        # common encodings (MS Word often uses Wingdings: ☒ checked, ☒/☑ vs ☐)
        if any(sym in t for sym in ["☑", "☒", "[x]", "[X]", "✓"]):
            m = CB_PAT.match(t)
            if m:
                checked.append(_norm(m.group(1)))
            else:
                # best-effort: strip known marks and keep suffix
                label = _norm(re.sub(r"^(?:☑|☒|\[x\]|\[X\]|✓)\s*", "", t))
                if label:
                    checked.append(label)
        elif "☐" in t or "[ ]" in t:
            label = _norm(t.replace("☐", "").replace("[ ]", ""))
            if label:
                unchecked.append(label)
    return checked, unchecked

# Optional: drawing-based checkbox detection (small square + X lines).
def _extract_checkboxes_drawn(page, clip_rect):
    """
    Heuristic: find small near-square drawings inside clip_rect that contain
    at least two diagonal-ish lines (an 'X'). Then read the label text to the right.
    Works on PyMuPDF 1.26.x get_drawings() output.
    """
    selected = []
    try:
        drawings = page.get_drawings()   # ✅ 1.26.x
    except Exception:
        return selected

    for d in drawings:
        r = d.get("rect")
        if not r:
            continue
        rect = fitz.Rect(r)
        if not rect.intersects(clip_rect):
            continue

        # candidate "checkbox" square
        w, h = rect.width, rect.height
        if w > 30 or h > 30 or abs(w - h) > 6:
            continue

        # count diagonal lines fully inside the square
        line_cnt = 0
        for it in d.get("items", []):
            coords = _line_coords_from_item(it)
            if not coords:
                continue
            x0, y0, x1, y1 = coords
            if rect.contains(fitz.Point(x0, y0)) and rect.contains(fitz.Point(x1, y1)):
                dx, dy = abs(x1 - x0), abs(y1 - y0)
                # diagonal-ish and long enough
                if dx > 2 and dy > 2 and min(dx, dy) / max(dx, dy) > 0.6:
                    line_cnt += 1

        if line_cnt >= 2:
            # grab text to the right as the label
            band = fitz.Rect(rect.x1 + 1, rect.y0 - 3, rect.x1 + 260, rect.y1 + 10) & clip_rect
            band_lines = _lines_from_clip(page, band)
            label = " ".join(_norm(L["text"]) for L in band_lines if _norm(L["text"]))
            if label:
                selected.append(_norm(label))

    return selected


def parse_office_via_pymupdf(input_path: str) -> dict:
    p = Path(input_path)
    pdf = office_to_pdf(p)         # now handles .docx, .pptx, or .pdf
    doc = fitz.open(str(pdf))
    pages = []
    for i, page in enumerate(doc):
        tables = _parse_tables_on_page(page)  # your existing function
        pages.append({"page": i + 1, "tables": tables})
    doc.close()
    return {"source": str(p), "derived_pdf": str(pdf), "pages": pages}


def _env_expand(s: str) -> str:
    return os.path.expandvars(s or "")

def _load_types_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    candidates = [config_path, os.environ.get("DOC_TYPES_CONFIG", None),
                  pathlib.Path(__file__).parent / "doc_types.json"]
    print("Looking for types config in:", candidates)
    for p in candidates:
        if p:
            p_str = str(p) if isinstance(p, pathlib.Path) else p
            if os.path.exists(p_str):
                try:
                    with open(p_str, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    return cfg
                except Exception as e:
                    print(f"[types-config] Failed to load {p_str}: {e}. ")
                    break



# ----------------------------- tiny utils ---------------------------------- #
SQUEEZE = lambda s: re.sub(r"[ \t\u00A0]+", " ", (s or "").strip())

def ensure_dirs(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def _guess_ext_from_magic(b: bytes) -> str:
    if b.startswith(b"\x89PNG\r\n\x1a\n"): return "png"
    if b.startswith(b"\xff\xd8"): return "jpg"
    if b[:6] in (b"GIF87a", b"GIF89a"): return "gif"
    if b.startswith(b"%PDF"): return "pdf"
    return "png"


# ------------------- NEW: classification + postprocess macro --------------- #
def _first_n_pages_json(pages_ordered, n=3, *, truncate_chars=300):
    """
    Build a compact JSON-like structure from the first n pages of `pages_ordered`.
    We keep only fields useful for coarse classification; texts are truncated to reduce prompt size.
    """
    pages = []
    for idx, elems in enumerate(pages_ordered[:max(0, n)], start=1):
        out_elems = []
        for b in elems:
            t = (b.get("text") or "")
            if truncate_chars and len(t) > truncate_chars:
                t = t[:truncate_chars] + " …"
            out_elems.append({
                "label": b.get("label","para"),
                "bbox": [int(round(v)) for v in (b.get("bbox") or [0,0,0,0])],
                "reading_order": int(b.get("reading_order", 0)),
                "text": t
            })
        pages.append({"page_number": idx, "elements": out_elems})
    return {"total_pages_sampled": len(pages), "pages": pages}

def _heuristic_classify(pages_ordered, n=3):
    """
    Offline fallback when ChatOllama isn't available.
    Heuristic: many images or short body text -> brochure; otherwise norm.
    """
    sample = pages_ordered[:n]
    total_blocks = 0
    image_like = 0
    text_chars = 0
    text_blocks = 0
    for elems in sample:
        for b in elems:
            total_blocks += 1
            lbl = (b.get("label") or "").lower()
            if lbl in ("fig","image"):
                image_like += 1
            else:
                t = (b.get("text") or "").replace(" ", "").replace("\n", "")
                if t:
                    text_blocks += 1
                    text_chars += len(t)
    img_ratio = (image_like / max(1, total_blocks))
    avg_chars = (text_chars / max(1, text_blocks))
    # thresholds tuned conservatively
    if img_ratio >= 0.35 or avg_chars < 80:
        return "brochure"
    return "norm"

def _parse_type_from_llm_text(txt: str, allowed: Iterable[str], synonyms: Optional[Dict[str, List[str]]] = None) -> str:
    if not txt: return ""
    # 先尝试“Type: xxx”格式
    m = re.search(r"Type\s*:\s*([A-Za-z]+)", txt, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip().lower()
        if t in {a.lower() for a in allowed}: return t
    # 其次在文本中直接找类型词或同义词
    low = txt.lower()
    allowed_lower = [a.lower() for a in allowed]
    for t in allowed_lower:
        if re.search(rf"\b{re.escape(t)}\b", low): return t
    if synonyms:
        for k, arr in synonyms.items():
            for syn in arr:
                if re.search(rf"\b{re.escape(syn.lower())}\b", low):
                    return k.lower()
    return ""

def _old_parse_type_from_llm_text(txt: str) -> str:
    """
    Parse 'Type: brochure' or 'Type: norm' from LLM output robustly.
    """
    if not txt:
        return ""
    m = re.search(r"Type\s*:\s*([A-Za-z]+)", txt, flags=re.IGNORECASE)
    if not m:
        # try simpler catch-all: just find 'brochure' / 'norm'
        if re.search(r"\bbrochure\b", txt, flags=re.IGNORECASE):
            return "brochure"
        if re.search(r"\bnorm\b", txt, flags=re.IGNORECASE):
            return "norm"
        return ""
    t = m.group(1).strip().lower()
    if t in ("brochure","norm"):
        return t
    return ""

def _import_callable(module_name: str, candidates, modules_dir: Optional[str] = None):
    """
    Try to import a module from a specific directory and return 
    the first callable attribute name in `candidates`.
    """
    import sys
    from pathlib import Path

    # Use specified directory or default to script's directory
    if modules_dir is None:
        modules_dir = str(Path(__file__).parent.resolve())
    else:
        modules_dir = str(Path(modules_dir).resolve())

    # Temporarily add the specific directory to sys.path
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)

    print(f"[DEBUG] Looking for module '{module_name}' in directory: {modules_dir}")

    try:
        # Force import from the specific directory
        spec = importlib.util.spec_from_file_location(
            module_name, 
            Path(modules_dir) / f"{module_name}.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Module '{module_name}' not found in '{modules_dir}'")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # Cache in sys.modules
        spec.loader.exec_module(mod)

    except FileNotFoundError:
        print(f"[DEBUG] Module file '{module_name}.py' not found in '{modules_dir}'")
        raise ImportError(f"Module '{module_name}' not found in '{modules_dir}'")
    except Exception as e:
        print(f"[DEBUG] Error loading module '{module_name}' from '{modules_dir}': {e}")
        raise

    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[DEBUG] Successfully loaded function '{name}' from module '{module_name}'")
            return fn

    raise AttributeError(f"Module '{module_name}' found but none of {candidates} are callable.")

def _default_norm_chunks(pages_subset):
    """
    Simple default postprocess for 'norm': linearize paragraphs/sections into ~800-char chunks.
    """
    PARA_LIKE = {"para","sec","sub_sec","sub_sub_sec"}
    buf, chunks, cur_len = [], [], 0
    def flush():
        nonlocal buf, chunks, cur_len
        if buf:
            text = "\n".join(buf).strip()
            if text:
                chunks.append({"type":"norm", "text": text})
            buf, cur_len = [], 0

    for page_idx, elems in enumerate(pages_subset, start=1):
        for e in elems:
            lbl = (e.get("label") or "").lower()
            if lbl in PARA_LIKE:
                t = e.get("text") or ""
                if not t.strip():
                    continue
                # greedily pack ~800 characters
                if cur_len + len(t) > 800:
                    flush()
                buf.append(t)
                cur_len += len(t)
    flush()
    return chunks

def classify_type_and_postprocess(
    pages_ordered,
    *,
    max_context_pages=3,
    ollama_base_url="http://localhost:11434",
    ollama_model="gpt-oss:20b",
    brochure_module: str = None,   # 向后兼容：作为候选优先项注入
    norm_module: str = None,       # 向后兼容：作为候选优先项注入
    types_config_path: Optional[str] = None,
    return_with_type: bool = True
):
    cfg = _load_types_config(types_config_path)
    allowed = cfg.get("allowed_types", []) or ["brochure","norm"]
    synonyms = cfg.get("synonyms", {}) or {}
    default_type = (cfg.get("default_type") or "rfp").lower()

    # 1) 上下文 JSON
    init_obj = _first_n_pages_json(pages_ordered, n=max_context_pages, truncate_chars=300)
    JSON_CONTEXT = json.dumps(init_obj, ensure_ascii=False, indent=2)

    # 2) 分类提示
    CLASSIFY_POSTFIX = (
        'According to the document parsing json file, what type the document is?\n\n'
        f'Answer in format: "Type: {{typename}}", typename is one of the following: {", ".join(allowed + ["None of them"])}'
    )
    prompt = JSON_CONTEXT + "\n\n" + CLASSIFY_POSTFIX
    # print('prompt: ', prompt)

    # 3) 调 LLM（失败则走启发式）
    doc_type = ""
    try:
        try:
            from langchain_community.chat_models import ChatOllama
        except Exception:
            from langchain_community.llms.ollama import Ollama as ChatOllama  # type: ignore
        base_url = ollama_base_url or os.environ.get("DEFAULT_OLLAMA", "http://localhost:11434")
        model = ollama_model or os.environ.get("DEFAULT_CHAT", "gpt-oss:20b")
        chat = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        allowed_for_prompt = ", ".join(allowed + ["None of them"])
        messages = [
            {"role": "system",
             "content": "You classify documents using the provided parsing JSON. "
                        f"Reply ONLY in the format: Type: <one of [{allowed_for_prompt}]>."},
            {"role": "user", "content": prompt},
        ]
        resp = chat.invoke(messages)
        llm_text = getattr(resp, "content", str(resp))
        doc_type = _parse_type_from_llm_text(llm_text, allowed, synonyms)
    except Exception as e:
        print(f"[classify] ChatOllama failed ({e}). Falling back to heuristic.")

    if not doc_type:
        # 启发式：仍只会给出 brochure/norm；若不在 allowed，则回退 default_type
        # heur = _heuristic_classify(pages_ordered, n=max_context_pages)
        doc_type = default_type


    # 4) 统一分发（读取配置）
    chunks = []
    tconf = (cfg.get("types", {}) or {}).get(doc_type, {})
    module_candidates = [ _env_expand(m) for m in (tconf.get("module_candidates") or []) ]
    module_candidates = [m for m in module_candidates if m and "${" not in m]

    # Add current directory to Python path to allow importing modules from the same directory
    current_dir = pathlib.Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    fn_candidates = (tconf.get("function_candidates") 
                     or cfg.get("function_candidates_default") 
                     or ["build_chunks"])

    called = False
    errors = []
    for modname in module_candidates:
        try:
            fn = _import_callable(modname, fn_candidates)
            print("----- Build chunks module FOUND ------", return_with_type)
            chunks = fn(pages_ordered)
            called = True
            break
        except Exception as ex:
            errors.append(f"{modname}: {ex}")
            continue

    if not called:
        fb = (tconf.get("fallback_builtin") or "default_norm").lower()
        print(f"[postprocess] No external module found for type '{doc_type}'. Fallback: {fb}")
        chunks = _default_norm_chunks(pages_ordered)

    return (doc_type, chunks) if return_with_type else chunks


def _old_classify_type_and_postprocess(
    pages_ordered,
    *,
    max_context_pages=3,
    ollama_base_url="http://localhost:11434",
    ollama_model="gpt-oss:20b",
    brochure_module: str = None,
    norm_module: str = None,
    return_with_type: bool = False
):
    """
    Macro:
      1) Take first `max_context_pages` from `pages_ordered`, serialize to JSON_CONTEXT.
      2) Build prompt: JSON_CONTEXT + CLASSIFY_POSTFIX
      3) Call ChatOllama to get: 'Type: brochure' or 'Type: norm'
      4) Parse typename and dispatch to specific postprocessors.
         - brochure: call functions defined in 'python file B' (module name may be passed via `brochure_module`).
         - norm: call module in `norm_module` if provided, else a reasonable default packer.

    Returns:
      - chunks (list[dict]) by default; if `return_with_type=True`, returns (doc_type, chunks).
    """
    # 1) Build JSON_CONTEXT
    init_obj = _first_n_pages_json(pages_ordered, n=max_context_pages, truncate_chars=300)
    JSON_CONTEXT = json.dumps(init_obj, ensure_ascii=False, indent=2)

    # 2) Prompt
    CLASSIFY_POSTFIX = (
        'According to the document parsing json file, what type the document is?\n\n'
        'Answer in format: "Type: {typename}", typename is one of the following: ["brochure", "norm", "RFP"]'
    )
    prompt = JSON_CONTEXT + "\n\n" + CLASSIFY_POSTFIX

    # 3) Call ChatOllama (best-effort)
    doc_type = ""
    llm_text = ""
    try:
        try:
            from langchain_community.chat_models import ChatOllama  # older LC
        except Exception:  # pragma: no cover
            from langchain_community.llms.ollama import Ollama as ChatOllama  # type: ignore
        base_url = ollama_base_url or os.environ.get("DEFAULT_OLLAMA", "http://localhost:11434")
        model = ollama_model or os.environ.get("DEFAULT_CHAT", "gpt-oss:20b")
        print('[debug]', model, base_url)
        chat = ChatOllama(model=model, base_url=base_url, temperature=0.0)
        messages = [
            {"role": "system",
             "content": "You classify documents using the provided parsing JSON. "
                        "Reply ONLY in the format: Type: brochure OR Type: norm."},
            {"role": "user", "content": prompt},
        ]
        resp = chat.invoke(messages)
        llm_text = getattr(resp, "content", str(resp))
        print('&***'*3, llm_text)
        doc_type = _parse_type_from_llm_text(llm_text)
    except Exception as e:
        print(f"[classify] ChatOllama not available or failed ({e}). Falling back to heuristic.")

    # Fallback if parsing failed
    if doc_type not in ("brochure","norm"):
        doc_type = _heuristic_classify(pages_ordered, n=max_context_pages)
        print(f"[classify] Heuristic decided: {doc_type}")

    # print(pages_ordered[:2])

    # 4) Dispatch
    chunks = []
    if doc_type == "brochure":
        # Try calling functions defined in "python file B"
        # You can set module name via `brochure_module` arg or env DOC_POST_B_MODULE
        module_candidates = []
        if brochure_module:
            module_candidates.append(brochure_module)
        module_candidates.extend([
            os.environ.get("DOC_POST_B_MODULE", "").strip() or "",
            "brochure_generate_chunks",
        ])
        module_candidates = [m for m in module_candidates if m]
        print(module_candidates)

        called = False
        errors = []
        for modname in module_candidates:
            try:
                fn = _import_callable(modname, [
                    "build_chunks",
                ])
                print(fn)
                chunks = fn(pages_ordered)
                called = True
                break
            except Exception as ex:
                errors.append(f"{modname}: {ex}")
                print(errors)
                continue
        if not called:
            print("[brochure] No external brochure postprocessor found; "
                  "falling back to default norm-style chunking.")
            chunks = _default_norm_chunks(pages_ordered)
    elif doc_type == "norm":
        # norm pipeline
        module_candidates = []
        if norm_module:
            module_candidates.append(norm_module)
        module_candidates.extend([
            os.environ.get("DOC_POST_B_MODULE", "").strip() or "",
            "norm_generate_chunks",
        ])
        module_candidates = [m for m in module_candidates if m]
        print(module_candidates)

        called = False
        errors = []
        for modname in module_candidates:
            try:
                fn = _import_callable(modname, [
                    "build_chunks",
                ])
                print(fn)
                chunks = fn(pages_ordered)
                called = True
                break
            except Exception as ex:
                errors.append(f"{modname}: {ex}")
                print(errors)
                continue
        if not called:
            print("[norm] No external norm postprocessor found; "
                  "falling back to default norm-style chunking.")
            chunks = _default_norm_chunks(pages_ordered)

    return (doc_type, chunks) if return_with_type else chunks


# ----------------------------- parsing: PyMuPDF ---------------------------- #
def _save_pixmap_crop(page, bbox, outpath, zoom=2.0):
    import fitz
    rect = fitz.Rect(bbox)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    pix.save(outpath)
    return outpath

def parse_with_pymupdf(pdf_path, outdir):
    """Return pages_blocks(list[list[block]]), page_sizes(list[(W,H)])"""
    import fitz
    import statistics
    doc = fitz.open(pdf_path)
    figures_dir = os.path.join(outdir, "figures")
    ensure_dirs(figures_dir)

    pages_blocks, page_sizes = [], []
    for pno in range(doc.page_count):
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        page_sizes.append((W, H))
        raw = page.get_text("dict")
        blocks = []
        img_counter = 0

        for blk in raw.get("blocks", []):
            btype = "text" if blk.get("type", 0) == 0 else ("image" if blk.get("type", 0) == 1 else "other")
            bbox = blk.get("bbox", [0, 0, 0, 0])

            if btype == "text":
                text_parts, max_sz, has_bold = [], 0.0, False
                for line in blk.get("lines", []):
                    spans = line.get("spans", [])
                    if spans:
                        text_parts.append("".join(s.get("text","") for s in spans))
                        span_sizes = [float(s["size"]) for s in spans if (s.get("text","") or "").strip()]  # 过滤纯空白
                        if span_sizes:                          
                            max_sz = max(max_sz, statistics.median(span_sizes))
                        if any(("Bold" in (s.get("font","") or "")) for s in spans): has_bold = True
                txt = SQUEEZE("\n".join(SQUEEZE(t) for t in text_parts if t is not None))
                if txt:
                    blocks.append({"type":"text","bbox":bbox,"text":txt,"font_size":max_sz,"has_bold":has_bold})

            elif btype == "image":
                if SAVE_IMAGE == False:
                    img_counter += 1
                    # 不生成图像文件，直接设为None
                    figure_path = None
                    # Markdown占位符保留（无实际文件）
                    md_text = "![Figure]()" if figure_path is None else f"![Figure]({figure_path})"
                    # 追加块信息（不含实际路径）
                    blocks.append({
                        "type": "image",
                        "bbox": bbox,
                        "text": md_text,
                        "figure_path": figure_path,
                        "label": "fig"
                    })
                elif SAVE_IMAGE == True:
                    # robust image extraction: inline bytes / xref / fallback crop
                    xref = None
                    figure_path = None
                    img_field = blk.get("image", None)
                    img_bytes_inline = bytes(img_field) if isinstance(img_field, (bytes, bytearray)) else None
                    if not img_bytes_inline:
                        if isinstance(img_field, dict) and "xref" in img_field:
                            xref = img_field["xref"]
                        elif "xref" in blk and isinstance(blk["xref"], int):
                            xref = blk["xref"]

                    img_counter += 1
                    base = f"{pathlib.Path(pdf_path).stem}_page_{pno+1:03d}_figure_{img_counter:03d}"
                    try:
                        if img_bytes_inline is not None:
                            ext = _guess_ext_from_magic(img_bytes_inline)
                            fpath = os.path.join(figures_dir, f"{base}.{ext}")
                            with open(fpath, "wb") as f: f.write(img_bytes_inline)
                            figure_path = os.path.join("markdown","figures",os.path.basename(fpath))
                        elif xref:
                            try:
                                img = doc.extract_image(xref)
                                ext = img.get("ext","png")
                                fpath = os.path.join(figures_dir, f"{base}.{ext}")
                                with open(fpath,"wb") as f: f.write(img["image"])
                                figure_path = os.path.join("markdown","figures",os.path.basename(fpath))
                            except Exception:
                                fpath = os.path.join(figures_dir, f"{base}_crop.png")
                                _save_pixmap_crop(page, bbox, fpath, zoom=2.0)
                                figure_path = os.path.join("markdown","figures",os.path.basename(fpath))
                        else:
                            fpath = os.path.join(figures_dir, f"{base}_crop.png")
                            _save_pixmap_crop(page, bbox, fpath, zoom=2.0)
                            figure_path = os.path.join("markdown","figures",os.path.basename(fpath))
                    except Exception:
                        fpath = os.path.join(figures_dir, f"{base}_pagecrop.png")
                        _save_pixmap_crop(page, page.rect, fpath, zoom=1.5)
                        figure_path = os.path.join("markdown","figures",os.path.basename(fpath))
                    md_text = f"![Figure]({figure_path})" if figure_path else "![Figure]()"
                    blocks.append({"type":"image","bbox":bbox,"text":md_text,"figure_path":figure_path,"label":"fig"})

        pages_blocks.append(blocks)
    doc.close()
    return pages_blocks, page_sizes

# ----------------------------- parsing: pdfplumber ------------------------- #
def _group_lines(words, y_tol=4):
    lines = []
    for w in sorted(words, key=lambda d: (d["top"], d["x0"])):
        placed = False
        for ln in lines:
            if abs(ln["top"]-w["top"])<=y_tol or abs(ln["bottom"]-w["bottom"])<=y_tol:
                ln["words"].append(w)
                ln["top"]=min(ln["top"],w["top"]); ln["bottom"]=max(ln["bottom"],w["bottom"])
                ln["x0"]=min(ln["x0"],w["x0"]); ln["x1"]=max(ln["x1"],w["x1"])
                placed=True; break
        if not placed:
            lines.append({"words":[w],"top":w["top"],"bottom":w["bottom"],"x0":w["x0"],"x1":w["x1"]})
    for ln in lines:
        ln["text"]=SQUEEZE(" ".join(w["text"] for w in sorted(ln["words"], key=lambda d: d["x0"])))
        ln["bbox"]=[ln["x0"],ln["top"],ln["x1"],ln["bottom"]]
    return sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

def _group_paragraphs(lines, line_gap_factor=1.6):
    if not lines: return []
    heights=[(ln["bbox"][3]-ln["bbox"][1]) for ln in lines]
    median_h=sorted(heights)[len(heights)//2]
    gap_thr=median_h * line_gap_factor
    blocks,cur=[],None
    prev_bottom=None
    for ln in lines:
        if cur is None:
            cur={"lines":[ln],"x0":ln["bbox"][0],"y0":ln["bbox"][1],"x1":ln["bbox"][2],"y1":ln["bbox"][3]}
            prev_bottom=ln["bbox"][3]; continue
        gap=ln["bbox"][1]-prev_bottom
        if gap>gap_thr:
            blocks.append(cur)
            cur={"lines":[ln],"x0":ln["bbox"][0],"y0":ln["bbox"][1],"x1":ln["bbox"][2],"y1":ln["bbox"][3]}
        else:
            cur["lines"].append(ln)
            cur["x0"]=min(cur["x0"],ln["bbox"][0]); cur["y0"]=min(cur["y0"],ln["bbox"][1])
            cur["x1"]=max(cur["x1"],ln["bbox"][2]); cur["y1"]=max(cur["y1"],ln["bbox"][3])
        prev_bottom=ln["bbox"][3]
    blocks.append(cur)
    out=[]
    for b in blocks:
        text="\n".join(ln["text"] for ln in b["lines"])
        out.append({"type":"text","bbox":[b["x0"],b["y0"],b["x1"],b["y1"]],"text":SQUEEZE(text),"font_size":0.0})
    return out

def parse_with_pdfplumber(pdf_path, outdir):
    import pdfplumber
    pages_blocks, page_sizes = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            W, H = p.width, p.height
            page_sizes.append((W,H))
            words = p.extract_words(use_text_flow=True, keep_blank_chars=False, x_tolerance=2, y_tolerance=2)
            lines = _group_lines(words, y_tol=4)
            blocks = _group_paragraphs(lines, line_gap_factor=1.6)
            pages_blocks.append(blocks)
    return pages_blocks, page_sizes

# --------------------------- doc-mode + order inference -------------------- #
def detect_document_mode(pages_blocks, page_sizes, *, cover_top_frac=0.28):
    def _w(b): x0,_,x1,_=b; return max(1.0, x1-x0)
    def _col_count(blocks,W):
        spans=[]
        for b in blocks:
            if (b.get("label") or "").lower() in ("foot","footer","cap","caption"): continue
            x0,_,x1,_=b["bbox"]; spans.append((x0,x1))
        spans.sort(); cols=[]
    
        for i, (x0,x1) in enumerate(spans):
            # print(i, x0, x1, cols, "  *******")
            placed=False
            for c in cols:
                if min(x1,c[1])-max(x0,c[0])>0.3*min(_w([x0,0,x1,0]),_w([c[0],0,c[1],0])):
                    c[0]=min(c[0],x0); c[1]=max(c[1],x1); placed=True; break
            if not placed: cols.append([x0,x1])
        print(len(cols))
        return len(cols)
    multi3=0
    for blocks,(W,H) in zip(pages_blocks,page_sizes):
        if _col_count(blocks,W)>=3: multi3+=1
    cover=False
    if pages_blocks:
        blocks=pages_blocks[0]; W,H=page_sizes[0]
        hi_titles=[b for b in blocks if b["bbox"][1] < cover_top_frac*H and (b["bbox"][2]-b["bbox"][0])>0.6*W]
        cover = len(hi_titles)>=1
    if multi3 >= max(1, len(pages_blocks)//3): return {"type":"poster","cover_pages":[0] if cover else []}
    if cover: return {"type":"regular_cover","cover_pages":[0]}
    return {"type":"regular","cover_pages":[]}


# --------------------------- 新增：单页列处理工具函数 ------------------------- #
def _get_page_col_count(blocks, page_width: float) -> int:
    """复用detect_document_mode中的_col_count逻辑，计算单页列数"""
    spans = []
    for b in blocks:
        # 过滤非正文块（页眉、页脚、图表标题等）
        if (b.get("label") or "").lower() in ("header", "footer", "cap", "caption"):
            continue
        x0, _, x1, _ = b["bbox"]  # 文本块水平边界
        spans.append((x0, x1))
    spans.sort()  # 按左边界排序
    cols = []
    # 合并重叠率≥30%的文本块范围，形成列
    for x0, x1 in spans:
        placed = False
        for c in cols:
            overlap = max(0.0, min(x1, c[1]) - max(x0, c[0]))
            min_width = max(1.0, min(x1 - x0, c[1] - c[0]))
            if overlap / min_width >= 0.30:  # 重叠率阈值
                c[0] = min(c[0], x0)
                c[1] = max(c[1], x1)
                placed = True
                break
        if not placed:
            cols.append([x0, x1])
    return len(cols)

def _get_page_col_ranges(blocks, page_width: float) -> List[List[float]]:
    """计算单页的列范围（每个列用[x0, x1]表示），基于文本块水平重叠"""
    spans = []
    for b in blocks:
        if (b.get("label") or "").lower() in ("header", "footer", "cap", "caption"):
            continue
        x0, _, x1, _ = b["bbox"]
        spans.append((x0, x1))
    spans.sort()
    cols = []
    # 合并重叠范围，得到列的原始范围
    for x0, x1 in spans:
        placed = False
        for c in cols:
            overlap = max(0.0, min(x1, c[1]) - max(x0, c[0]))
            min_width = max(1.0, min(x1 - x0, c[1] - c[0]))
            if overlap / min_width >= 0.30:
                c[0] = min(c[0], x0)
                c[1] = max(c[1], x1)
                placed = True
                break
        if not placed:
            cols.append([x0, x1])
    # 按左边界排序，确保列顺序从左到右
    return sorted(cols, key=lambda c: c[0])

def infer_reading_order_strategy(pages_blocks, page_sizes, doc_mode, *,
                                 top_band_frac=0.01, edge_band_frac=0.10,
                                 multi_col_threshold=2, flip_majority=0.60):
    def _detect_cols(blocks,W):
        xs=[]; 
        for b in blocks:
            if (b.get("label") or "").lower() in ("header","footer","cap","caption"): continue
            x0,_,x1,_=b["bbox"]; xs.append((x0,x1))
        xs.sort(); cols=[]
        # print('+++**'*3, H3 != doc_state['body_font'])

        for x0,x1 in xs:
            placed=False
            for c in cols:
                inter=max(0.0,min(x1,c[1])-max(x0,c[0])); minw=max(1.0,min(x1-x0,c[1]-c[0]))
                if inter/minw>=0.30: c[0]=min(c[0],x0); c[1]=max(c[1],x1); placed=True; break
            if not placed: cols.append([x0,x1])
        return cols
    
    def _first_body_side(blocks,W,H):
        top_cut=top_band_frac*H
        body=[b for b in blocks if (b.get("label","").lower() not in ("header","footer","cap","caption") and b["bbox"][1]<=top_cut)]
        if not body: return None
        b=sorted(body,key=lambda bb:(bb["bbox"][1],bb["bbox"][0]))[0]
        cx=0.5*(b["bbox"][0]+b["bbox"][2])
        return "left" if cx<(W/2) else "right"
    
    def _edge_bias(blocks,W,H):
        lband=edge_band_frac*W; rband=(1-edge_band_frac)*W
        L=R=0
        for b in blocks:
            x0,y0,x1,y1=b["bbox"]; area=max(1.0,(x1-x0)*(y1-y0))
            if area>0.25*W*H: continue
            if x0<=lband: L+=1
            if x1>=rband: R+=1
        if L>R*1.2: return +1
        if R>L*1.2: return -1
        return 0

    per_page_cols = {}  # 键：page_idx，值：该页的列数
    for page_idx, (blocks, (W, H)) in enumerate(zip(pages_blocks, page_sizes)):
        per_page_cols[page_idx] = _get_page_col_count(blocks, W)

    # ------------------- 新增：打印每页列数的debug代码 -------------------
    print("\n[Debug] 每页列数统计：")
    for page_idx in sorted(per_page_cols.keys()):
        # 页码从1开始显示（符合用户习惯），page_idx是0-based
        print(f"  第{page_idx + 1}页：{per_page_cols[page_idx]}列")

    mode=(doc_mode.get("type") or "regular").lower()
    cover_pages=set(doc_mode.get("cover_pages",[]))
    prior="ltr"  # default per your spec
    right_starts=left_starts=0; multi=0; edge_votes=0
    for blocks,(W,H) in zip(pages_blocks,page_sizes):
        cols=_detect_cols(blocks,W)
        if len(cols)>=multi_col_threshold:
            multi+=1
            side=_first_body_side(blocks,W,H)
            if side=="right": right_starts+=1
            elif side=="left": left_starts+=1
        edge_votes+=_edge_bias(blocks,W,H)
    decided=prior
    if multi>=1:
        if right_starts >= flip_majority*max(1,right_starts+left_starts):
            decided="rtl"
        elif left_starts >= flip_majority*max(1,right_starts+left_starts):
            decided="ltr"
        else:
            decided = "ltr" if edge_votes>=0 else "rtl"
    per_page={}
    if mode=="regular_cover":
        for i in cover_pages: per_page[i]="ltr"
    # return {"col_order": decided, "per_page_override": per_page}
    return {
        "col_order": decided,  # 文档整体阅读方向（LTR/RTL）
        "per_page_override": per_page,  # 单页阅读方向覆盖
        "per_page_cols": per_page_cols  # 新增：每页的列数
    }
# ------------------


# --------------------------- global state + layout ------------------------- #
def build_global_state(pages_blocks, page_sizes, *,
                       prefer_col_order="ltr",
                       header_frac=0.12, footer_frac=0.10,
                       heading_quantiles=(1,0.88,0.78),
                       max_cols=3, gutter_shrink=0.04):
    import numpy as np
    from collections import defaultdict
    def _ensure_fs(b):
        fs=float(b.get("font_size",0.0))
        if fs>0: return fs
        x0,y0,x1,y1=b.get("bbox",[0,0,0,0]); lines=max(1,(b.get("text","") or "").count("\n")+1)
        return (y1-y0)/lines if y1>y0 else 0.0

    # 步骤1：定义需要排除的非正文标签（小写）
    excluded_labels = {"header", "footer", "cap", "caption", "image", "fig", "table"}

    # 2. 定位中间三页（优先分析文档核心内容页）
    total_pages = len(pages_blocks)
    if total_pages <= 3:
        middle_pages = list(range(total_pages))
    else:
        mid = total_pages // 2
        middle_pages = [mid - 1, mid, mid + 1]
        middle_pages = [max(0, min(p, total_pages - 1)) for p in middle_pages]
        middle_pages = list(sorted(set(middle_pages)))

    # # 3. 收集中间三页的正文block字体大小及字符数
    # fs_counts = defaultdict(int)  # 字体大小→出现的block次数
    # fs_char_counts = defaultdict(int)  # 字体大小→对应总字符数
    # for p_idx in middle_pages:
    #     blocks = pages_blocks[p_idx]
    #     for b in blocks:
    #         blk_label = (b.get("label") or "").lower()
    #         if blk_label in excluded_labels:
    #             continue  # 过滤非正文
    #         fs = _ensure_fs(b)
    #         if fs <= 0:
    #             continue  # 跳过无效字体大小
    #         fs_counts[fs] += 1
    #         char_count = len(b.get("text", "").replace(" ", "").replace("\n", ""))
    #         fs_char_counts[fs] += char_count

    # # 4. 确定body_font
    # body_font = 12.0  # 默认值
    # if fs_counts:
    #     sorted_by_block = sorted(fs_counts.items(), key=lambda x: (-x[1], -fs_char_counts[x[0]]))
    #     body_font = sorted_by_block[0][0]
    # 3. 收集中间三页的正文block字体大小及字符数
    fs_counts = defaultdict(int)  # 字体大小→出现的block次数
    fs_char_counts = defaultdict(int)  # 字体大小→对应总字符数
    for p_idx in middle_pages:
        blocks = pages_blocks[p_idx]
        for b in blocks:
            blk_label = (b.get("label") or "").lower()
            if blk_label in excluded_labels:
                continue  # 过滤非正文
            fs = _ensure_fs(b)
            if fs <= 0:
                continue  # 跳过无效字体大小
            fs_counts[fs] += 1  # 统计block次数（备用）
            char_count = len(b.get("text", "").replace(" ", "").replace("\n", ""))
            fs_char_counts[fs] += char_count  # 统计字符总数（核心）

    # 4. 确定body_font：按字符总数排序（优先），字符数相同再按block次数
    body_font = 12.0  # 默认值
    if fs_char_counts:  # 若有字符数统计
        # 排序规则：先按字符总数降序，再按block次数降序
        sorted_by_chars = sorted(fs_char_counts.items(), 
                                key=lambda x: (-x[1], -fs_counts.get(x[0], 0)))
        body_font = round(sorted_by_chars[0][0], 2)  # 取字符数最多的字体大小



    print("**-"*3, "middle_pages: ", middle_pages, sorted_by_chars, "body font: ", body_font)

    fs_all=[]
    for blocks,_ in zip(pages_blocks,page_sizes):
        for b in blocks:
            blk_label = (b.get("label") or "").lower()
            if blk_label in excluded_labels:
                continue
            fs_all.append(_ensure_fs(b))  
    if not fs_all: 
        fs_all=[12.0]

    fs_all = [float(f) for f in fs_all if f is not None]
    fs_all = [round(f, 2) for f in fs_all]
    unique_fs = sorted(set(fs_all), reverse=True)
    while len(unique_fs) < 3:
        unique_fs.append(unique_fs[-1])
    qH1, qH2, qH3 = unique_fs[0], unique_fs[1], unique_fs[2]

    print('>--'*5, unique_fs, '~@'*3, qH1, qH2, qH3)

    avgH = sum(h for _,h in page_sizes)/max(1,len(page_sizes))
    header_y1 = header_frac * avgH
    footer_y0 = (1.0 - footer_frac) * avgH

    # global columns
    spans=[]
    for blocks,(W,H) in zip(pages_blocks,page_sizes):
        for b in blocks:
            y0=b["bbox"][1]
            if y0<header_y1: continue
            if (b.get("label") or "").lower() in ("foot","footer","cap","caption"): continue
            spans.append((b["bbox"][0], b["bbox"][2]))
    spans.sort(); cols=[]
    for x0,x1 in spans:
        placed=False
        for c in cols:
            inter=max(0.0,min(x1,c["x1"])-max(x0,c["x0"]))
            minw=max(1.0,min(x1-x0,c["x1"]-c["x0"]))
            if inter/minw>=0.30:
                c["x0"]=min(c["x0"],x0); c["x1"]=max(c["x1"],x1); placed=True; break
        if not placed: cols.append({"x0":x0,"x1":x1})
        if len(cols)>=max_cols: break
    if cols:
        meanW = sum(w for w,_ in page_sizes)/max(1,len(page_sizes))
        shrink = gutter_shrink * meanW
        cols = sorted(cols, key=lambda c: c["x0"])
        for i,c in enumerate(cols):
            c["span"]=[max(0.0,c["x0"]+shrink), c["x1"]-shrink]; c["index"]=i
        col_indices = list(range(len(cols)))
        if prefer_col_order=="rtl": col_indices = list(reversed(col_indices))
    else:
        col_indices=[0]

    return {
        "font_quantiles":{"H1":qH1,"H2":qH2,"H3":qH3},
        "header_y1": header_y1,
        "footer_y0": footer_y0,
        "body_font": body_font,  
        "columns": [{"index":c["index"],"span":c["span"]} for c in cols],
        "col_order": prefer_col_order,
        "col_indices_ordered": col_indices,
    }

# === DROP-IN REPLACEMENT ===
def apply_layout_to_document(pages_blocks, page_sizes, doc_state, order_info):
    def _upper_ratio(t):
        letters=[c for c in t if c.isalpha()]
        return (sum(c.isupper() for c in letters)/len(letters)) if letters else 0.0
    def _ensure_fs(b):
        fs = float(b.get("font_size", 0.0))
        return fs if fs > 0 else ( (b["bbox"][3]-b["bbox"][1]) / max(1, (b.get("text","") or "").count("\n")+1) )
    def _assign_block_to_col(block, col_ranges: List[List[float]]) -> int:
        x0, _, x1, _ = block["bbox"]
        max_overlap = -1.0
        col_idx = 0
        for i, (c_x0, c_x1) in enumerate(col_ranges):
            overlap = max(0.0, min(x1, c_x1) - max(x0, c_x0))
            if overlap > max_overlap:
                max_overlap = overlap
                col_idx = i
        return col_idx

    H1=doc_state["font_quantiles"]["H1"]
    H2=doc_state["font_quantiles"]["H2"]
    H3=doc_state["font_quantiles"]["H3"]
    header_y1=doc_state["header_y1"]
    footer_y0=doc_state["footer_y0"]
    columns=doc_state["columns"]
    col_ids=doc_state["col_indices_ordered"]
    body_font = float(doc_state.get("body_font", 12.0))

    out=[]
    for page_idx, (blocks, (W, H)) in enumerate(zip(pages_blocks, page_sizes)):
        blocks = deepcopy(blocks)

        # 1) classify + label (now with 'sideinfo' support)
        for b in blocks:
            fs=_ensure_fs(b)
            y0=b["bbox"][1]
            lbl=(b.get("label") or "").lower()
            txt=b.get("text","") or ""

            if y0<=header_y1 and fs < body_font:
                b["level"]=0; b["label"]="header"
                continue
            if y0>=footer_y0:
                b["level"]=5; b["label"]="footer"
                continue

            # heading levels by font quantiles
            if fs>=H1 and H1 != body_font: level=1
            elif fs>=H2 and H2 != body_font: level=2
            elif fs>=H3 and H3 != body_font: level=3
            else: level=4

            # all-caps slightly promotes heading
            if _upper_ratio(txt.replace("\n"," "))>=0.65:
                level=max(1,level-1)

            # figures/captions guardrails
            if lbl in ("fig","image"): level=max(level,3)
            if lbl in ("cap","caption"): level=min(level,4)

            # --- NEW: sideinfo detection (font strictly smaller than body_font - 0.5) ---
            if lbl not in ("fig","image","cap","caption","header","footer"):
                if fs < (body_font - 0.5):
                    b["level"] = 4
                    b["label"] = "sideinfo"
                else:
                    b["level"] = level
                    b["label"] = {1:"sec",2:"sub_sec",3:"sub_sub_sec",4:"para",5:"footer"}.get(level,"para")
            else:
                b["level"] = level

        # 2) per-page column ranges
        page_col_ranges = _get_page_col_ranges(blocks, W)

        # 3) split header/footer/body
        header = [b for b in blocks if b["label"] == "header"]
        footer = [b for b in blocks if b["label"] == "footer"]
        body =   [b for b in blocks if b["label"] not in ("header","footer")]

        # 4) assign explicit col
        for b in header + footer:
            b["col"] = -1

        if (order_info["per_page_cols"].get(page_idx, 1) >= 2) and page_col_ranges:
            col_blocks = {i: [] for i in range(len(page_col_ranges))}
            for b in body:
                col_idx = _assign_block_to_col(b, page_col_ranges)
                b["col"] = col_idx
                col_blocks[col_idx].append(b)
            def _col_sort_key(b):
                y = b["bbox"][1]
                is_heading = b["label"] in ("sec","sub_sec","sub_sub_sec")
                return (round(y/2), 0 if is_heading else 1, b["bbox"][0])
            sorted_body=[]
            for col_idx in sorted(col_blocks.keys()):
                sorted_body.extend(sorted(col_blocks[col_idx], key=_col_sort_key))
        else:
            def _single_col_sort_key(b):
                y = b["bbox"][1]
                is_heading = b["label"] in ("sec","sub_sec","sub_sub_sec")
                return (round(y/2), 0 if is_heading else 1, b["bbox"][0])
            sorted_body = sorted(body, key=_single_col_sort_key)
            for b in body:
                b["col"] = 0

        ordered=[]
        ordered.extend(sorted(header, key=lambda b: (b["bbox"][1], b["bbox"][0])))
        ordered.extend(sorted_body)
        ordered.extend(sorted(footer, key=lambda b: (b["bbox"][1], b["bbox"][0])))

        for i, b in enumerate(ordered):
            b["reading_order"] = i
        out.append(ordered)
    return out


def old_apply_layout_to_document(pages_blocks, page_sizes, doc_state, order_info):
    def _upper_ratio(t):
        letters=[c for c in t if c.isalpha()]
        return (sum(c.isupper() for c in letters)/len(letters)) if letters else 0.0
    def _ensure_fs(b):
        fs = float(b.get("font_size", 0.0))
        return fs if fs > 0 else ( (b["bbox"][3]-b["bbox"][1]) / max(1, (b.get("text","") or "").count("\n")+1) )
    
    def _assign_block_to_col(block, col_ranges: List[List[float]]) -> int:
        """将文本块分配到对应的列（基于x坐标重叠最大的列）"""
        x0, _, x1, _ = block["bbox"]
        max_overlap = -1.0
        col_idx = 0  # 默认第0列
        for i, (c_x0, c_x1) in enumerate(col_ranges):
            overlap = max(0.0, min(x1, c_x1) - max(x0, c_x0))
            if overlap > max_overlap:
                max_overlap = overlap
                col_idx = i
        return col_idx
    
    def _overlap_w(a0,a1,b0,b1): return max(0.0, min(a1,b1)-max(a0,b0))
    def _assign_col(b, columns):
        bx0,_,bx1,_=b["bbox"]; best=-1; score=0.0
        for c in columns:
            cx0,cx1=c["span"]; s=_overlap_w(bx0,bx1,cx0,cx1)/max(1.0,(bx1-bx0))
            if s>score: score=s; best=c["index"]
        return best

    H1=doc_state["font_quantiles"]["H1"]; 
    H2=doc_state["font_quantiles"]["H2"]; 
    H3=doc_state["font_quantiles"]["H3"]
    header_y1=doc_state["header_y1"]; 
    footer_y0=doc_state["footer_y0"]
    columns=doc_state["columns"]; 
    col_ids=doc_state["col_indices_ordered"]

    out=[]
    for page_idx, (blocks, (W, H)) in enumerate(zip(pages_blocks, page_sizes)):
        blocks = deepcopy(blocks)
        for b in blocks:
            fs=_ensure_fs(b); y0=b["bbox"][1]; lbl=(b.get("label") or "").lower(); txt=b.get("text","") or ""
            if y0<=header_y1 and fs < doc_state["body_font"]:
                b["level"]=0; b["label"]="header"
            elif y0>=footer_y0:
                b["level"]=5; b["label"]="footer"
            else:
                if fs>=H1 and H1 != doc_state["body_font"]: level=1
                elif fs>=H2 and H2 != doc_state["body_font"]: level=2
                elif fs>=H3 and H3 != doc_state["body_font"]: level=3
                else: level=4
                if _upper_ratio(txt.replace("\n"," "))>=0.65: level=max(1,level-1)
                if lbl in ("fig","image"): level=max(level,3)
                if lbl in ("cap","caption"): level=min(level,4)
                b["level"]=level
                if lbl not in ("fig","image","cap","caption"):
                    b["label"]={1:"sec",2:"sub_sec",3:"sub_sub_sec",4:"para",5:"footer"}.get(level,"para")

        # 2. 获取当前页的列数和列范围（核心新增逻辑）
        page_col_count = order_info["per_page_cols"].get(page_idx, 1)  # 从order_info获取列数
        page_col_ranges = _get_page_col_ranges(blocks, W)  # 计算当前页的列范围（左到右）

        # 3. 分离页眉、页脚、正文块
        header = [b for b in blocks if b["label"] == "header"]
        footer = [b for b in blocks if b["label"] == "footer"]
        body = [b for b in blocks if b["label"] not in ("header", "footer")]

        # 4. 为所有块分配col属性（核心修复逻辑）
        # 4.1 页眉和页脚：固定col=-1（非正文列）
        for b in header + footer:
            b["col"] = -1  # 明确赋值

        # 4. 正文块按列分配并排序（核心改动）
        if page_col_count >= 2 and page_col_ranges:  # 多列处理
            # 按列分组
            col_blocks = {i: [] for i in range(len(page_col_ranges))}
            for b in body:
                col_idx = _assign_block_to_col(b, page_col_ranges)
                b["col"] = col_idx  # 明确赋值列索引（0,1,2...）
                col_blocks[col_idx].append(b)
            # 列内排序：先按y（上下），再按x（左右），标题优先
            def _col_sort_key(b):
                y = b["bbox"][1]
                is_heading = b["label"] in ("sec", "sub_sec", "sub_sub_sec")
                return (round(y / 2), 0 if is_heading else 1, b["bbox"][0])  # y容错2像素
            # 按LTR顺序拼接列（左→右）
            sorted_body = []
            for col_idx in sorted(col_blocks.keys()):  # 确保列顺序从左到右
                sorted_body.extend(sorted(col_blocks[col_idx], key=_col_sort_key))
        else:  # 单列处理：直接按y→x排序
            def _single_col_sort_key(b):
                y = b["bbox"][1]
                is_heading = b["label"] in ("sec", "sub_sec", "sub_sub_sec")
                return (round(y / 2), 0 if is_heading else 1, b["bbox"][0])
            sorted_body = sorted(body, key=_single_col_sort_key)
            for b in body:
                b["col"] = 0  # 单列默认col=0

        # 5. 组合页眉、正文、页脚，生成最终阅读顺序
        ordered = []
        # 页眉排序（上下→左右）
        ordered.extend(sorted(header, key=lambda b: (b["bbox"][1], b["bbox"][0])))
        # 正文（已按列排序）
        ordered.extend(sorted_body)
        # 页脚排序（上下→左右）
        ordered.extend(sorted(footer, key=lambda b: (b["bbox"][1], b["bbox"][0])))

        # 6. 分配reading_order序号
        for i, b in enumerate(ordered):
            b["reading_order"] = i
        out.append(ordered)

    return out

# --------------------------- debug renderer -------------------------------- #
def render_layout_debug_pdf(pdf_path, pages_blocks, page_sizes, doc_state, order_info, out_pdf_path=None):
    import fitz

    def _ensure_fs(b):
        fs = float(b.get("font_size", 0.0))
        if fs > 0:
            return fs, 0
        # fallback: line-height proxy
        x0,y0,x1,y1 = b.get("bbox", [0,0,0,0])
        text = (b.get("text") or "")
        n_lines = max(1, text.count("\n") + 1)
        return (y1 - y0) / n_lines, 1 if y1 > y0 else 0.0
    
    out_pdf_path = out_pdf_path or str(pathlib.Path(pdf_path).with_name(pathlib.Path(pdf_path).stem + "_smart_debug.pdf"))
    doc = fitz.open(pdf_path)
    assert len(doc)==len(pages_blocks)==len(page_sizes)
    pages_ordered = apply_layout_to_document(pages_blocks, page_sizes, doc_state, order_info)

    for pno, page in enumerate(doc, start=1):
        W,H = page.rect.width, page.rect.height
        elements = pages_ordered[pno-1]
        columns = doc_state["columns"]
        # columns bands
        for col in columns:
            x0,x1 = col["span"]
            page.draw_rect(fitz.Rect(x0, 0, x1, H), fill=(1,1,0.6), fill_opacity=0.12, overlay=True)
            page.insert_text((x0+4, 10), f"col {col['index']}", fontsize=8, color=(0,0,0), overlay=True)
        level_color = {0:(0,0,0),1:(1,0,0),2:(1,0.5,0),3:(0.2,0.5,1),4:(0,0.45,0),5:(0.5,0.5,0.5)}
        for b in elements:
            x0,y0,x1,y1=b["bbox"]
            fs_vis, fs_code = _ensure_fs(b)
            page.draw_rect(fitz.Rect(x0,y0,x1,y1),
                           color=level_color.get(b.get("level",3),(0,0,0)),
                           width=0.8,
                           fill=(1,1,0.85),
                           fill_opacity=0.25,
                           stroke_opacity=0.9,
                           overlay=True)
            tag = f"#{b.get('reading_order','?')} L{b.get('level','?')} col:{b.get('col','?')} {b.get('label','')} fs:{fs_vis:.3f} | {fs_code}"
            page.insert_text((x0+2, y0+9), tag, fontsize=7.5, color=(0,0,0), overlay=True)
    doc.save(out_pdf_path); doc.close()
    return out_pdf_path

# --------------------------- JSON writer ----------------------------------- #
def write_init_json(pdf_path, pages_ordered, out_json):
    data = {
        "source_file": os.path.abspath(pdf_path),
        "total_pages": len(pages_ordered),
        "pages": []
    }
    for idx, elems in enumerate(pages_ordered, start=1):
        out_elems=[]
        for b in elems:
            e={"label": b.get("label","para"),
               "bbox": [int(round(v)) for v in b["bbox"]],
               "text": b.get("text",""),
               "reading_order": int(b.get("reading_order",0))}
            if e["label"]=="fig" and b.get("figure_path"):
                e["figure_path"]=b["figure_path"]
            out_elems.append(e)
        data["pages"].append({"page_number": idx, "elements": out_elems})
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    return out_json

def write_json(pdf_path, pages_ordered, out_json):
    data = {
        "source_file": os.path.abspath(pdf_path),
        "total_pages": len(pages_ordered),
        "pages": []
    }
    for idx, elems in enumerate(pages_ordered, start=1):
        out_elems=[]
        for b in elems:
            col_val = b.get("col", -1)
            e={"label": b.get("label","para"),
               "bbox": [int(round(v)) for v in b["bbox"]],
               "text": b.get("text",""),
               "reading_order": int(b.get("reading_order",0)),
               "col": col_val  # 关键：添加col信息到JSON输出
               }
            if e["label"]=="fig" and b.get("figure_path"):
                e["figure_path"]=b["figure_path"]
            out_elems.append(e)
        data["pages"].append({"page_number": idx, "elements": out_elems})
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    return out_json

def write_chunks_json(chunks, out_path):
    ensure_dirs(pathlib.Path(out_path).parent.as_posix())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    return out_path


# ----------------------------- public API ---------------------------------- #
def create_chunks_coarsetofine(
    pdf_path: str,
    *,
    backend: str = "pymupdf",
    brochure_module: Optional[str] = None,   # optional custom postprocessor module with build_chunks(...)
    norm_module: Optional[str] = None,       # optional custom postprocessor module with build_chunks(...)
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parse `pdf_path` → infer structure → classify → produce chunks.
    Returns a list of chunk dicts with at least:
        text, page, bbox, orig_size, (optional) type/reading_order/label
    """
    from typing import List, Dict, Any, Optional
    import os
    import pathlib
    from copy import deepcopy

    ori_pdf_path = os.path.abspath(pdf_path)
    file_ext = os.path.splitext(ori_pdf_path)[1].lower()

    if file_ext == '.pptx':
        out = pptx_to_pdf_win(ori_pdf_path)
        pdf_path = out
    elif file_ext == ".docx":
        out = docx_to_pdf(ori_pdf_path)
        pdf_path = out
    else:
        pdf_path = ori_pdf_path  

    TAG = "[c2f-api]"
    # pdf_path = os.path.abspath(pdf_path)
    if verbose:
        print(f"{TAG} Processing PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 初始化输出目录（用于临时存储图片等资源）
    outdir = pathlib.Path(pdf_path).with_suffix("").as_posix() + "_assets"
    figures_dir = os.path.join(outdir, "figures")
    ensure_dirs(figures_dir)

    

    # 1. 解析PDF（根据backend选择引擎）
    pages_blocks, page_sizes = [], []
    if backend == "pymupdf":
        if verbose:
            print(f"{TAG} Using PyMuPDF backend for parsing")
        pages_blocks, page_sizes = parse_with_pymupdf(pdf_path, outdir)
    elif backend == "pdfplumber":
        if verbose:
            print(f"{TAG} Using pdfplumber backend for parsing")
        pages_blocks, page_sizes = parse_with_pdfplumber(pdf_path, outdir)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'pymupdf' or 'pdfplumber'")

    if not pages_blocks:
        if verbose:
            print(f"{TAG} No content parsed from PDF")
        return []

    # 2. 文档模式检测（封面、海报等）
    doc_mode = detect_document_mode(pages_blocks, page_sizes)
    if verbose:
        print(f"{TAG} Detected document mode: {doc_mode['type']}")

    # 3. 推断阅读顺序策略（含每页列数）
    order_info = infer_reading_order_strategy(pages_blocks, page_sizes, doc_mode)
    if verbose:
        print(f"{TAG} Reading order strategy: {order_info['col_order']}")
        for page_idx in sorted(order_info["per_page_cols"].keys()):
            print(f"{TAG} Page {page_idx + 1} columns: {order_info['per_page_cols'][page_idx]}")

    # 4. 构建全局状态（字体、页眉页脚位置、列信息等）
    doc_state = build_global_state(
        pages_blocks,
        page_sizes,
        prefer_col_order=order_info["col_order"],
        header_frac=0.12,
        footer_frac=0.10
    )

    # 5. 应用布局（确定阅读顺序、块级别、列归属）
    pages_ordered = apply_layout_to_document(pages_blocks, page_sizes, doc_state, order_info)
    

    json_out = pathlib.Path(pdf_path).with_name(f"{pathlib.Path(pdf_path).stem}_initchunks.json").as_posix()
    write_init_json(pdf_path, pages_ordered, json_out)
    print("Wrote JSON:", json_out)


    # 6. 分类文档类型并生成 chunks（优先外部模块，默认按字符数打包）
    if verbose:
        print(f"{TAG} Starting document classification and chunking")
    doc_type, chunks = classify_type_and_postprocess(
        pages_ordered,
        max_context_pages=3,
        brochure_module=brochure_module,
        norm_module=norm_module,
        return_with_type=True
    )

    print(f"Classified Type: {doc_type}")
    pdf_path_obj = pathlib.Path(pdf_path)
    chunks_out = pdf_path_obj.with_name(f"{pdf_path_obj.stem}_pubapi_chunks.json").as_posix()
    outp = write_chunks_json(chunks, chunks_out)
    print("Chunks JSON:", outp)


    if verbose:
        print(f"{TAG} Generated {len(chunks)} valid chunks")

    return chunks

def _main():
    ap = argparse.ArgumentParser(description="PDF → structured JSON with global layout + debug PDF + (optional) classify+postprocess")
    ap.add_argument("pdf", help="Path to input PDF")
    args = ap.parse_args()
    pdf_path = args.pdf
    _ = create_chunks_coarsetofine(pdf_path)
# --------------------------- CLI ------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="PDF → structured JSON with global layout + debug PDF + (optional) classify+postprocess")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--backend", choices=["pymupdf","pdfplumber"], default="pymupdf")
    ap.add_argument("--outdir", default="markdown", help="Base output dir (for figures)")
    ap.add_argument("--json_out", default=None, help="Output JSON path")
    ap.add_argument("--debug_pdf", default=None, help="Output annotated PDF path")
    # New flags
    ap.add_argument("--classify_postprocess", default=True, help="Run classify_type_and_postprocess on first 3 pages")
    ap.add_argument("--chunks_out", default=None, help="If set, write resulting chunks to this JSON file")
    ap.add_argument("--ollama_base_url", default="http://localhost:11434")
    ap.add_argument("--ollama_model", default="gpt-oss:20b")
    ap.add_argument("--brochure_module", default=None, help="Module name for brochure postprocess (python file B)")
    ap.add_argument("--norm_module", default=None, help="Module name for norm postprocess")
    args = ap.parse_args()

    ori_pdf_path = args.pdf
    ensure_dirs(args.outdir)
    file_ext = os.path.splitext(ori_pdf_path)[1].lower()

    if file_ext == '.pptx':
        out = pptx_to_pdf_win(ori_pdf_path)
        pdf_path = out
    else:
        pdf_path = ori_pdf_path  

    if args.backend=="pymupdf":
        pages_blocks, page_sizes = parse_with_pymupdf(pdf_path, args.outdir)
    else:
        pages_blocks, page_sizes = parse_with_pdfplumber(pdf_path, args.outdir)

    # Pre-phase: mode + reading order
    doc_mode = detect_document_mode(pages_blocks, page_sizes)
    order_info = infer_reading_order_strategy(pages_blocks, page_sizes, doc_mode)

    # Global state + layout
    doc_state = build_global_state(
        pages_blocks, page_sizes,
        prefer_col_order=order_info["col_order"],
        header_frac=0.12, footer_frac=0.10,
        heading_quantiles=(0.98,0.9,0.78)
    )
    # 修改：调用apply_layout_to_document时传入order_info
    pages_ordered = apply_layout_to_document(pages_blocks, page_sizes, doc_state, order_info)

    # # Per-page overrides (e.g., cover)
    # for i, override in order_info["per_page_override"].items():
    #     local = dict(doc_state)
    #     local["col_indices_ordered"] = (list(range(len(doc_state["columns"]))) if override=="ltr"
    #                                     else list(reversed(range(len(doc_state["columns"])))))
    #     pages_ordered[i] = apply_layout_to_document([pages_blocks[i]], [page_sizes[i]], local)[0]

    # JSON
    json_out = args.json_out or (pathlib.Path(pdf_path).with_suffix(".json").as_posix())
    write_json(pdf_path, pages_ordered, json_out)
    print("Wrote JSON:", json_out, pdf_path)

    # Debug PDF
    dbg_out = args.debug_pdf or str(pathlib.Path(pdf_path).with_name(pathlib.Path(pdf_path).stem + "_smart_debug.pdf"))
    render_layout_debug_pdf(pdf_path, pages_blocks, page_sizes, doc_state, order_info, dbg_out)
    print("Debug PDF:", dbg_out)

    # Optional: classify + postprocess
    if args.classify_postprocess:
        doc_type, chunks = classify_type_and_postprocess(
            pages_ordered,
            max_context_pages=3,
            ollama_base_url=args.ollama_base_url,
            ollama_model=args.ollama_model,
            brochure_module=args.brochure_module,
            norm_module=args.norm_module,
            return_with_type=True
        )
        print(f"Classified Type: {doc_type}")
        pdf_path_obj = pathlib.Path(pdf_path)
        chunks_out = args.chunks_out or pdf_path_obj.with_name(f"{pdf_path_obj.stem}_chunks.json").as_posix()
        outp = write_chunks_json(chunks, chunks_out)
        print("Chunks JSON:", outp)

if __name__ == "__main__":
    main()
