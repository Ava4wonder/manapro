# backend/src/tender_analyzer/apps/api_gateway/routes/tenders.py
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from tender_analyzer.apps.ingestion.workers.ingestion_worker import process_tender_and_store_in_qdrant
from tender_analyzer.domain.dto import QuestionAnswerDTO, SummaryResponse as SummaryResponseDTO
from tender_analyzer.domain.models import QuestionAnswer, StoredDocument, Tender
from tender_analyzer.domain.repositories import tender_repo
from tender_analyzer.common.state.enums import TenderState

from tender_analyzer.apps.qa_analysis.answer_pipeline import run_summary_analysis


router = APIRouter()
LOG = logging.getLogger(__name__)

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

SUMMARY_READY_STATES = {
    TenderState.SUMMARY_READY,
    TenderState.FULL_RUNNING,
    TenderState.FULL_READY,
    TenderState.EVAL_RUNNING,
    TenderState.EVAL_READY,
}

def _status_payload(tender: Tender):
    """Convert tender object to API response"""
    state = tender.state
    return {
        "id": tender.id,
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
    }

SUMMARY_PROMPT = (
    "Summarize the key requirements, risks, and deliverables described in the uploaded tender. "
    "Highlight timeline expectations, mandatory qualifications, and anything that would help a reviewer understand the scope."
)


import json
import logging
from pathlib import Path
from typing import List

from tender_analyzer.apps.qa_analysis.answer_pipeline import run_summary_analysis
from tender_analyzer.common.state.state_machine import TenderStateMachine
from tender_analyzer.common.state.enums import TenderState
from tender_analyzer.domain.repositories import tender_repo

# 假设 QuestionAnswer 实体仍用于其他部分，此处仅为展示
# from tender_analyzer.domain.entities import QuestionAnswer

LOG = logging.getLogger(__name__)

def _run_summary_pipeline_task(tender_id: str) -> None:
    """
    执行摘要分析流程，并将结果以 JSONL 字符串格式存储。
    """
    LOG.info("[analysis] Running summary pipeline for tender %r", tender_id)
    
    # 初始化状态机用于状态转移验证
    state_machine = TenderStateMachine()
    
    try:
        # 1. 运行摘要分析 pipeline
        # 假设 run_summary_analysis 返回一个字典，其中包含 'output_file' 键，其值为生成的 JSONL 文件路径
        pipeline_result = run_summary_analysis(tender_id)
        
        # 2. 验证并获取输出文件路径
        output_file_path = pipeline_result.get("output_file")
        if not output_file_path:
            raise ValueError("Summary pipeline did not return a valid 'output_file' path.")
        
        output_path = Path(output_file_path)
        if not output_path.exists() or not output_path.is_file():
            raise FileNotFoundError(f"Summary output file not found at: {output_path}")

        LOG.info("[analysis] Successfully generated summary file for tender %r at %s", tender_id, output_path)

        # 3. 读取 JSONL 文件内容作为字符串
        with open(output_path, "r", encoding="utf-8") as f:
            jsonl_content = f.read()
        
        # 4. 验证并更新 Tender 状态
        # 这是一个重要的健壮性检查，确保状态转移符合预期
        tender = tender_repo.get(tender_id)
        if not tender:
            raise ValueError(f"Tender with ID {tender_id} not found in repository.")
        
        # 检查当前状态是否允许转移到 SUMMARY_READY
        # 从 SUMMARY_RUNNING 应该能转移到 SUMMARY_READY
        current_state = tender.state
        LOG.info("[analysis] Current state of tender %r: %s", tender_id, current_state)
        
        # 验证状态转移的有效性
        if not state_machine.can_transition(current_state, TenderState.SUMMARY_READY):
            raise ValueError(f"Invalid transition from {current_state} to {TenderState.SUMMARY_READY}")
        
        # 5. 更新 Tender 的 highlight_answers 字段 (现在是 JSONL 字符串)
        # 注意：这里我们假设 tender_repo.update_highlight_answers 方法已被修改为接受一个字符串
        # 如果方法名或签名不同，请根据实际情况调整
        tender_repo.update_highlight_answers(tender_id, jsonl_content)
        
        # 6. 将 Tender 状态设置为 SUMMARY_READY
        tender_repo.set_state(tender_id, TenderState.SUMMARY_READY)
        
        LOG.info("[analysis] Tender %r summary ready. JSONL content stored.", tender_id)

    except Exception as e:
        LOG.exception("[analysis] Summary pipeline failed for tender %r", tender_id)
        try:
            # 尝试将状态设置为 FAILED
            tender = tender_repo.get(tender_id)
            if tender:
                current_state = tender.state
                # 即使在失败的情况下，也应尝试进行合法的状态转移
                if state_machine.can_transition(current_state, TenderState.FAILED):
                    tender_repo.set_state(tender_id, TenderState.FAILED)
                else:
                    # 如果状态转移失败（例如，当前状态已不是允许转移的状态），仍强制设置为 FAILED 作为最后的手段
                    LOG.warning("[analysis] Cannot transition from %s to FAILED. Forcing state change anyway.", current_state)
                    tender_repo.set_state(tender_id, TenderState.FAILED)
        except Exception:
            LOG.critical("[analysis] Failed to set tender %r state to FAILED.", tender_id)

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


@router.get("/tenders/{tender_id}/summary")
async def get_summary(tender_id: str):
    """Return stored highlight answers for a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")

    questions = [
        QuestionAnswerDTO(question=answer.question, answer=answer.answer)
        for answer in tender.highlight_answers
    ]

    return SummaryResponseDTO(
        id=tender.id,
        questions=questions,
        ready=tender.state in SUMMARY_READY_STATES,
    )


@router.post("/tenders/{tender_id}/start-analysis")
async def start_analysis(tender_id: str, background_tasks: BackgroundTasks):
    """Start analysis for a tender"""
    tender = tender_repo.get(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="tender not found")
    
    # Update state to indicate analysis is running
    tender_repo.set_state(tender_id, TenderState.SUMMARY_RUNNING)

    background_tasks.add_task(_run_summary_pipeline_task, tender_id)
    
    return _status_payload(tender)
