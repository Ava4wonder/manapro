import uuid
from datetime import datetime
from typing import List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):
        return None

from tender_analyzer.common.state.enums import TenderState

SQLALCHEMY_AVAILABLE = False
try:
    from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.orm import relationship
    from tender_analyzer.common.db.base import Base
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    Column = String = DateTime = Integer = Boolean = ForeignKey = lambda *args, **kwargs: None  # type: ignore
    UUID = str
    relationship = lambda *args, **kwargs: None  # type: ignore
    Base = object


if SQLALCHEMY_AVAILABLE:
    class User(Base):
        __tablename__ = "users"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        tenant_id = Column(UUID(as_uuid=True), nullable=False)
        email = Column(String, unique=True, nullable=False)
        role = Column(String, nullable=False, default="user")
        is_active = Column(Boolean, default=True)


    class LoginCode(Base):
        __tablename__ = "login_codes"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
        code = Column(String, nullable=False)
        created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
        expires_at = Column(DateTime, nullable=False)
        consumed_at = Column(DateTime, nullable=True)
        attempt_count = Column(Integer, nullable=False, default=0)

        user = relationship("User")


    class Document(Base):
        __tablename__ = "documents"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        tenant_id = Column(UUID(as_uuid=True), nullable=False)
        tender_id = Column(UUID(as_uuid=True), nullable=False)
        filename = Column(String, nullable=False)
        storage_path = Column(String, nullable=False)
else:
    class User:
        pass

    class LoginCode:
        pass

    class Document:
        pass

class StoredDocument(BaseModel):
    id: str
    name: str
    storage_path: str
    uploaded_at: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class Evaluation(BaseModel):
    summary: str
    risk_level: str
    recommendation: str


class Tender(BaseModel):
    id: str
    name: str
    tenant_id: str
    created_at: str
    state: TenderState
    documents: List[StoredDocument] = Field(default_factory=list)
    highlight_answers: str = ""  # str for storing JSONL
    full_answers: List[QuestionAnswer] = Field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    analysis_corpus: str = ""

    class Config:
        validate_assignment = True
