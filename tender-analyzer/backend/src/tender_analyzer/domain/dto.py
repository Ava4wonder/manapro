from typing import List, Optional

from pydantic import BaseModel

from tender_analyzer.domain.models import Evaluation, QuestionAnswer


class QuestionAnswerDTO(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    status: Optional[str] = None
    processing_time_sec: Optional[float] = None
    error_message: Optional[str] = None


class SummaryResponse(BaseModel):
    id: str
    questions: List[QuestionAnswerDTO]
    ready: bool


class DetailsResponse(BaseModel):
    id: str
    questions: List[QuestionAnswerDTO]
    ready: bool


class EvaluationResponse(BaseModel):
    id: str
    evaluation: Optional[Evaluation]


class StatusResponse(BaseModel):
    id: str
    state: str
    summary_ready: bool
    full_ready: bool
    eval_ready: bool
