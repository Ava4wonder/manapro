# session
# backend/src/tender_analyzer/common/db/session.py

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from tender_analyzer.common.config.settings import settings
from tender_analyzer.common.db.base import Base  # <-- import Base
from tender_analyzer.domain import models        # <-- ensure models are imported so tables are registered

from tender_analyzer.common.config.settings import settings


# Example: DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/tenders"
# Put this in your .env and load via settings
DATABASE_URL = settings.DATABASE_URL  # make sure this exists in settings.py

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    future=True,        # SQLAlchemy 1.4+ style
    echo=False,         # set True if you want SQL logs in dev
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    Usage in routes:
        @router.get("/something")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables based on ORM models.

    Dev-only / simple bootstrap. In production, prefer migrations (Alembic).
    """
    Base.metadata.create_all(bind=engine)