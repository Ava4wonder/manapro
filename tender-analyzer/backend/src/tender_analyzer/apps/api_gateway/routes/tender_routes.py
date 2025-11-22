# from typing import Iterable, List

# from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

# from tender_analyzer.apps.evaluation.service import EvaluationService
# from tender_analyzer.apps.ingestion.service import ProcessingService
# from tender_analyzer.apps.orchestration.service import OrchestrationService
# from tender_analyzer.apps.qa_engine.service import AnalysisService
# from tender_analyzer.common.auth.middleware import get_current_user
# from tender_analyzer.common.auth.models import AuthenticatedUser
# from tender_analyzer.common.state.enums import TenderState
# from tender_analyzer.domain.models import QuestionAnswer
# from tender_analyzer.domain.repositories import tender_repo

# router = APIRouter()

# processing_service = ProcessingService(tender_repo)
# analysis_service = AnalysisService()
# evaluation_service = EvaluationService()
# orchestration_service = OrchestrationService(
#     repository=tender_repo,
#     analysis_service=analysis_service,
#     evaluation_service=evaluation_service,
# )

# STATE_PROGRESS = {
#     TenderState.INGESTING: 5,
#     TenderState.INGESTED: 20,
#     TenderState.SUMMARY_RUNNING: 40,
#     TenderState.SUMMARY_READY: 60,
#     TenderState.FULL_RUNNING: 70,
#     TenderState.FULL_READY: 85,
#     TenderState.EVAL_RUNNING: 90,
#     TenderState.EVAL_READY: 100,
#     TenderState.FAILED: 0,
# }


# def _answers_to_payload(answers: Iterable[QuestionAnswer]) -> List[dict]:
#     return [{"question": answer.question, "answer": answer.answer} for answer in answers]


# def _must_get(tender_id: str, user: AuthenticatedUser):
#     tender = tender_repo.get(tender_id)
#     if not tender or tender.tenant_id != user.tenant_id:
#         raise HTTPException(status_code=404, detail="tender not found")
#     return tender


# def _status_payload(tender):
#     state = tender.state
#     return {
#         "id": tender.id,
#         "state": state.value,
#         "progress": STATE_PROGRESS.get(state, 0),
#         "summary_ready": state in {
#             TenderState.SUMMARY_READY,
#             TenderState.FULL_RUNNING,
#             TenderState.FULL_READY,
#             TenderState.EVAL_RUNNING,
#             TenderState.EVAL_READY,
#         },
#         "full_ready": state in {
#             TenderState.FULL_READY,
#             TenderState.EVAL_RUNNING,
#             TenderState.EVAL_READY,
#         },
#         "eval_ready": state == TenderState.EVAL_READY,
#         "documents": [document.name for document in tender.documents],
#         "created_at": tender.created_at,
#     }


# @router.post("/tenders")
# async def create_tender(
#     name: str = Form(...),
#     files: List[UploadFile] | None = File(None),
#     user: AuthenticatedUser = Depends(get_current_user),
# ):
#     tender = await processing_service.upload_package(name, user.tenant_id, files)
#     return {"id": tender.id}


# @router.get("/tenders/{tender_id}/status")
# async def get_status(tender_id: str, user: AuthenticatedUser = Depends(get_current_user)):
#     tender = _must_get(tender_id, user)
#     return _status_payload(tender)


# @router.post("/tenders/{tender_id}/start-analysis")
# async def start_analysis(tender_id: str, user: AuthenticatedUser = Depends(get_current_user)):
#     _must_get(tender_id, user)
#     try:
#         orchestration_service.start_analysis(tender_id)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))
#     tender = _must_get(tender_id, user)
#     return _status_payload(tender)


# @router.get("/tenders/{tender_id}/summary")
# async def get_summary(tender_id: str, user: AuthenticatedUser = Depends(get_current_user)):
#     tender = _must_get(tender_id, user)
#     return {
#         "id": tender.id,
#         "ready": tender.state in {
#             TenderState.SUMMARY_READY,
#             TenderState.FULL_RUNNING,
#             TenderState.FULL_READY,
#             TenderState.EVAL_RUNNING,
#             TenderState.EVAL_READY,
#         },
#         "questions": _answers_to_payload(tender.highlight_answers),
#     }


# @router.get("/tenders/{tender_id}/details")
# async def get_details(tender_id: str, user: AuthenticatedUser = Depends(get_current_user)):
#     tender = _must_get(tender_id, user)
#     return {
#         "id": tender.id,
#         "ready": tender.state in {
#             TenderState.FULL_READY,
#             TenderState.EVAL_RUNNING,
#             TenderState.EVAL_READY,
#         },
#         "questions": _answers_to_payload(tender.full_answers),
#     }


# @router.get("/tenders/{tender_id}/evaluation")
# async def get_evaluation(tender_id: str, user: AuthenticatedUser = Depends(get_current_user)):
#     tender = _must_get(tender_id, user)
#     return {
#         "id": tender.id,
#         "ready": tender.state == TenderState.EVAL_READY,
#         "evaluation": tender.evaluation.dict() if tender.evaluation else None,
#     }


# backend/src/tender_analyzer/apps/api_gateway/routes/tenders.py
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List
import uuid
from pathlib import Path
from datetime import datetime

from tender_analyzer.apps.ingestion.workers.ingestion_worker import process_tender_and_store_in_qdrant
from tender_analyzer.domain.repositories import tender_repo
from tender_analyzer.domain.models import Tender, StoredDocument
from tender_analyzer.common.state.enums import TenderState

from tender_analyzer.apps.qa_analysis.answer_pipeline import run_summary_analysis


router = APIRouter()

# 临时文件存储目录
TEMP_DIR = Path(__file__).parent.parent.parent / "storage" / "tender_uploads"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# STATE_PROGRESS mapping for status endpoint
STATE_PROGRESS = {
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

def _status_payload(tender: Tender):
    """Convert tender object to API response"""
    state = tender.state
    return {
        "id": tender.id,
        "state": state.value,
        "progress": STATE_PROGRESS.get(state, 0),
        "summary_ready": state in {
            TenderState.SUMMARY_READY,
            TenderState.FULL_RUNNING,
            TenderState.FULL_READY,
            TenderState.EVAL_RUNNING,
            TenderState.EVAL_READY,
        },
        "full_ready": state in {
            TenderState.FULL_READY,
            TenderState.EVAL_RUNNING,
            TenderState.EVAL_READY,
        },
        "eval_ready": state == TenderState.EVAL_READY,
        "documents": [doc.name for doc in tender.documents],
        "created_at": tender.created_at,
    }

@router.post("/tenders")
async def upload_tender(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # 生成唯一的 tender ID
    tender_id = str(uuid.uuid4())
    
    # 创建初始 Tender 对象并存储在仓库中（状态为 INGESTING）
    tender = Tender(
        id=tender_id,
        name=name,
        tenant_id="default-tenant",  # TODO: get from auth context
        state=TenderState.INGESTING,
        created_at=datetime.utcnow().isoformat(),
        documents=[]
    )
    tender_repo.create(tender)
    
    # 创建临时目录存储上传的文件
    tender_dir = TEMP_DIR / tender_id
    tender_dir.mkdir(exist_ok=True)
    
    # 保存上传的文件到临时目录
    for file in files:
        file_path = tender_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
    
    # 触发异步任务：分块 + Qdrant 存储
    background_tasks.add_task(
        process_tender_and_store_in_qdrant,
        tender_id=tender_id,
        tender_dir=str(tender_dir),
        name=name
    )
    
    # 返回 tender ID 给前端
    return {"id": tender_id}

@router.get("/tenders/{tender_id}/status")
async def get_status(tender_id: str):
    """Get the status of a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")
    return _status_payload(tender)

@router.post("/tenders/{tender_id}/start-analysis")
async def start_analysis(tender_id: str):
    """Start analysis for a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")
    
    # Update state to indicate analysis is running
    tender_repo.set_state(tender_id, TenderState.SUMMARY_RUNNING)
    
    return _status_payload(tender)