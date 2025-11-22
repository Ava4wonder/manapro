from pydantic_settings import BaseSettings
from typing import List



class Settings(BaseSettings):
    app_name: str = "Tender Analyzer API"
    default_tenant: str = "tenant_default"
    api_prefix: str = "/api"

    # ---- JWT auth ----
    JWT_SECRET_KEY: str = "change-me-in-prod"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRES_MINUTES: int = 60 * 24  # 24 hours
    DATABASE_URL: str = "sqlite:///./dev.db"  # or Postgres URL, etc.

    # ---- Allowed login domains ----
    # Comma-separated list in .env, e.g. "gruner.ch,example.com"
    ALLOWED_LOGIN_EMAIL_DOMAINS: str = "gruner.ch"

    # ---- Vector store (Qdrant) ----
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_VECTOR_DIM: int = 64
    QDRANT_USE_GRPC: bool = False


    class Config:
        env_file = ".env"


settings = Settings()

def get_allowed_domains() -> List[str]:
    # Helper: turn ALLOWED_LOGIN_EMAIL_DOMAINS string -> list
    raw = settings.ALLOWED_LOGIN_EMAIL_DOMAINS
    return [d.strip().lower() for d in raw.split(",") if d.strip()]
