# backend/src/tender_analyzer/apps/api_gateway/routes/tenders.py
import json
import logging
import mimetypes
import shutil
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from tender_analyzer.apps.ingestion.workers.ingestion_worker import process_tender_and_store_in_qdrant
from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.common.state.state_machine import TenderStateMachine
from tender_analyzer.domain.dto import QuestionAnswerDTO, SummaryResponse as SummaryResponseDTO
from tender_analyzer.domain.models import QuestionAnswer, StoredDocument, Tender
from tender_analyzer.domain.repositories import tender_repo

from tender_analyzer.apps.qa_analysis.answer_pipeline import run_summary_analysis


router = APIRouter()
LOG = logging.getLogger(__name__)

# Upload directory root
TEMP_DIR = Path(__file__).parent.parent.parent / "storage" / "tender_uploads"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# STATE_PROGRESS mapping for status endpoint
STATE_PROGRESS = {
    TenderState.PENDING: 0,
    TenderState.INGESTING: 5,
    TenderState.INGESTED: 20,
    TenderState.SUMMARY_RUNNING: 40,
    TenderState.SUMMARY_READY: 60,
    TenderState.FULL_RUNNING: 70,
    TenderState.FULL_READY: 85,
    TenderState.EVAL_RUNNING: 90,
    TenderState.EVAL_READY: 100,
    TenderState.FAILED: 0,
}

SUMMARY_READY_STATES = {
    TenderState.SUMMARY_READY,
    TenderState.FULL_RUNNING,
    TenderState.FULL_READY,
    TenderState.EVAL_RUNNING,
    TenderState.EVAL_READY,
}


def _status_payload(tender: Tender) -> Dict[str, Any]:
    """Convert tender object to API response"""
    state = tender.state
    return {
        "id": tender.id,
        "name": tender.name,
        "state": state.value,
        "progress": STATE_PROGRESS.get(state, 0),
        "summary_ready": state in SUMMARY_READY_STATES,
        "full_ready": state in {
            TenderState.FULL_READY,
            TenderState.EVAL_RUNNING,
            TenderState.EVAL_READY,
        },
        "eval_ready": state == TenderState.EVAL_READY,
        "documents": [doc.name for doc in tender.documents],
        "created_at": tender.created_at,
        "project_fields": getattr(tender, "project_card_fields", {}),
    }


SUMMARY_PROMPT = (
    "Summarize the key requirements, risks, and deliverables described in the uploaded tender. "
    "Highlight timeline expectations, mandatory qualifications, and anything that would help a reviewer understand the scope."
)


def _run_summary_pipeline_task(tender_id: str, tenant_id: str) -> None:
    """
    Run the summary pipeline for a tender and persist the JSONL results.
    """
    LOG.info("[analysis] Running summary pipeline for tender %r", tender_id)

    state_machine = TenderStateMachine()

    try:
        # 1. Run the pipeline
        pipeline_result = run_summary_analysis(tender_id, tenant_id)

        # 2. Validate output file
        output_file_path = pipeline_result.get("output_file")
        if not output_file_path:
            raise ValueError("Summary pipeline did not return a valid 'output_file' path.")

        output_path = Path(output_file_path)
        if not output_path.exists() or not output_path.is_file():
            raise FileNotFoundError(f"Summary output file not found at: {output_path}")

        LOG.info(
            "[analysis] Successfully generated summary file for tender %r at %s",
            tender_id,
            output_path,
        )

        # 3. Read JSONL content
        with output_path.open("r", encoding="utf-8") as f:
            jsonl_content = f.read()

        # 4. Load tender from repository
        tender = tender_repo.get(tender_id)
        if not tender:
            raise ValueError(f"Tender with ID {tender_id} not found in repository.")

        current_state = tender.state
        LOG.info("[analysis] Current state of tender %r: %s", tender_id, current_state)

        # Ensure we can move from SUMMARY_RUNNING -> SUMMARY_READY
        if not state_machine.can_transition(current_state, TenderState.SUMMARY_READY):
            raise ValueError(f"Invalid transition from {current_state} to {TenderState.SUMMARY_READY}")

        # 5. Persist JSONL to tender.highlight_answers
        tender_repo.update_highlight_answers(tender_id, jsonl_content)

        # 6. Mark summary as ready
        tender_repo.set_state(tender_id, TenderState.SUMMARY_READY)

        LOG.info("[analysis] Tender %r summary ready. JSONL content stored.", tender_id)

    except Exception:
        LOG.exception("[analysis] Summary pipeline failed for tender %r", tender_id)
        try:
            tender = tender_repo.get(tender_id)
            if tender:
                current_state = tender.state
                if state_machine.can_transition(current_state, TenderState.FAILED):
                    tender_repo.set_state(tender_id, TenderState.FAILED)
                else:
                    LOG.warning(
                        "[analysis] Cannot transition from %s to FAILED. Forcing state change anyway.",
                        current_state,
                    )
                    tender_repo.set_state(tender_id, TenderState.FAILED)
        except Exception:
            LOG.critical("[analysis] Failed to set tender %r state to FAILED.", tender_id)


# ---------------------------------------------------------------------------
# Simple in-process FIFO queue for tender processing
# ---------------------------------------------------------------------------

_QUEUE_LOCK: Lock = Lock()
_PROCESS_QUEUE: "deque[Dict[str, str]]" = deque()
_ACTIVE_JOB: Optional[Dict[str, str]] = None


def _start_next_job_unlocked() -> None:
    """Start the next queued tender if no job is active."""
    global _ACTIVE_JOB
    if _ACTIVE_JOB is not None:
        return
    if not _PROCESS_QUEUE:
        return

    job = _PROCESS_QUEUE.popleft()
    _ACTIVE_JOB = job

    def _run_pipeline_job(payload: Dict[str, str]) -> None:
        global _ACTIVE_JOB
        tender_id = payload["tender_id"]
        tender_dir = payload["tender_dir"]
        name = payload["name"]
        tenant_id = payload["tenant_id"]

        LOG.info("[queue] Starting queued job for tender %s", tender_id)
        try:
            # Mark tender as INGESTING if possible
            tender = tender_repo.get(tender_id)
            if tender:
                state_machine = TenderStateMachine()
                current_state = tender.state
                if state_machine.can_transition(current_state, TenderState.INGESTING):
                    tender_repo.set_state(tender_id, TenderState.INGESTING)

            # Step 1: ingest + store chunks in Qdrant
            process_tender_and_store_in_qdrant(
                tender_id=tender_id,
                tender_dir=tender_dir,
                name=name,
                tenant_id=tenant_id,
            )

            # Step 2: transition to SUMMARY_RUNNING and run summary pipeline
            tender = tender_repo.get(tender_id)
            if not tender:
                LOG.warning("[queue] Tender %s disappeared before summary step", tender_id)
            else:
                state_machine = TenderStateMachine()
                current_state = tender.state
                if state_machine.can_transition(current_state, TenderState.SUMMARY_RUNNING):
                    tender_repo.set_state(tender_id, TenderState.SUMMARY_RUNNING)
                    _run_summary_pipeline_task(tender_id, tenant_id)
                else:
                    LOG.warning(
                        "[queue] Cannot move tender %s from %s to SUMMARY_RUNNING; skipping summary",
                        tender_id,
                        current_state,
                    )
        except Exception:
            LOG.exception("[queue] Pipeline job failed for tender %s", tender_id)
        finally:
            with _QUEUE_LOCK:
                _ACTIVE_JOB = None
                _start_next_job_unlocked()

    Thread(target=_run_pipeline_job, args=(job,), daemon=True).start()


def _enqueue_tender_for_processing(
    tender_id: str,
    tender_dir: str,
    name: str,
    tenant_id: str,
) -> None:
    """
    Enqueue a tender for sequential ingest + summary processing.
    The first queued tender runs immediately; subsequent ones wait
    until the previous reaches SUMMARY_READY / FAILED.
    """
    job: Dict[str, str] = {
        "tender_id": tender_id,
        "tender_dir": tender_dir,
        "name": name,
        "tenant_id": tenant_id,
    }

    with _QUEUE_LOCK:
        # Avoid enqueuing duplicates for the same tender_id.
        if _ACTIVE_JOB and _ACTIVE_JOB.get("tender_id") == tender_id:
            return
        if any(j.get("tender_id") == tender_id for j in _PROCESS_QUEUE):
            return

        _PROCESS_QUEUE.append(job)
        _start_next_job_unlocked()


def _build_question_answer_dto(record: Any) -> Optional[QuestionAnswerDTO]:
    if isinstance(record, QuestionAnswer):
        payload = record.dict()
    elif isinstance(record, dict):
        payload = record
    else:
        return None
    
    dto = QuestionAnswerDTO(
        question=str(payload.get("question") or ""),
        answer=str(payload.get("answer") or ""),
        category=payload.get("category"),
        subcategory=payload.get("subcategory"),
        status=payload.get("status"),
        processing_time_sec=payload.get("processing_time_sec"),
        error_message=payload.get("error_message"),
        references=payload.get("references") or [],
    )

    # DEBUG: sample refs
    try:
        from logging import getLogger
        LOG = getLogger(__name__)
        first_refs = dto.references[:3] if dto.references else []
        LOG.debug("Built QuestionAnswerDTO with %d references; sample=%s",
                  len(dto.references), first_refs)
    except Exception:
        pass

    return dto


def _parse_highlight_answers(value: Any) -> List[QuestionAnswerDTO]:
    records: List[QuestionAnswerDTO] = []

    def push(item: Any) -> None:
        dto = _build_question_answer_dto(item)
        if dto:
            records.append(dto)

    if isinstance(value, (list, tuple)):
        for item in value:
            push(item)
        return records

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value or "")

    text = text.strip()
    if not text:
        return records

    parsed: Any | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            push(item)
        if records:
            return records
    elif isinstance(parsed, dict):
        push(parsed)
        if records:
            return records

    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            candidate = json.loads(trimmed)
        except json.JSONDecodeError:
            continue
        push(candidate)

    return records


@router.get("/tenders")
async def list_tenders():
    """Return all tenders for project cards."""
    tenders = tender_repo.list()
    return [_status_payload(t) for t in tenders]


@router.post("/tenders")
async def upload_tender(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Create a new tender and enqueue it for processing."""
    tender_id = str(uuid.uuid4())

    tender = Tender(
        id=tender_id,
        name=name,
        tenant_id="default-tenant",  # TODO: get from auth context
        state=TenderState.PENDING,
        created_at=datetime.utcnow().isoformat(),
        documents=[],
    )
    tender_repo.create(tender)

    tender_dir = TEMP_DIR / tender_id
    tender_dir.mkdir(exist_ok=True)

    for file in files:
        file_path = tender_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        tender.documents.append(
            StoredDocument(
                id=str(uuid.uuid4()),
                name=file.filename,
                storage_path=str(file_path),
                uploaded_at=datetime.utcnow().isoformat(),
            )
        )

    # Enqueue for sequential ingest + summary processing
    _enqueue_tender_for_processing(
        tender_id=tender_id,
        tender_dir=str(tender_dir),
        name=name,
        tenant_id=tender.tenant_id,
    )

    return {"id": tender_id}


@router.get("/tenders/{tender_id}/status")
async def get_status(tender_id: str):
    """Get the status of a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")
    return _status_payload(tender)


@router.get("/tenders/{tender_id}/summary")
async def get_summary(tender_id: str):
    """Return stored highlight answers for a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")

    questions = _parse_highlight_answers(tender.highlight_answers)

    return SummaryResponseDTO(
        id=tender.id,
        questions=questions,
        ready=tender.state in SUMMARY_READY_STATES,
    )


@router.get("/tenders/{tender_id}/documents/{document_name:path}")
async def download_tender_document(tender_id: str, document_name: str):
    """Stream back the requested tender document so the UI can render PDF previews."""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")

    for doc in tender.documents:
        if doc.name == document_name:
            storage_path = Path(doc.storage_path)
            if not storage_path.exists():
                raise HTTPException(status_code=404, detail="document file missing on server")
            media_type, _ = mimetypes.guess_type(doc.name)
            return FileResponse(
                path=str(storage_path),
                media_type=media_type or "application/octet-stream",
                filename=doc.name,
            )

    raise HTTPException(status_code=404, detail="document not found for this tender")


@router.post("/tenders/{tender_id}/start-analysis")
async def start_analysis(tender_id: str, background_tasks: BackgroundTasks):
    """Start analysis for a tender (manual re-run)."""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")

    state_machine = TenderStateMachine()
    current_state = tender.state
    if not state_machine.can_transition(current_state, TenderState.SUMMARY_RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start analysis from state '{current_state.value}'. "
            f"Valid state: {TenderState.INGESTED.value}",
        )

    tender_repo.set_state(tender_id, TenderState.SUMMARY_RUNNING)

    background_tasks.add_task(_run_summary_pipeline_task, tender_id, tender.tenant_id)

    return _status_payload(tender)


@router.delete("/tenders/{tender_id}", status_code=204)
async def delete_tender(tender_id: str):
    """Delete a tender, its uploaded files, and remove it from the repository."""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")

    tender_dir = TEMP_DIR / tender_id
    if tender_dir.exists():
        shutil.rmtree(tender_dir, ignore_errors=True)

    # TODO: Optionally delete associated vectors from Qdrant

    tender_repo.delete(tender_id)
