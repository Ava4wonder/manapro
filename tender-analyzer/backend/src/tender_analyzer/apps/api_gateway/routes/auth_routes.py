# from datetime import datetime

# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, EmailStr, constr

# from tender_analyzer.common.auth.service import auth_service

# router = APIRouter()


# class RequestCodeRequest(BaseModel):
#     email: EmailStr


# class RequestCodeResponse(BaseModel):
#     email: EmailStr
#     message: str
#     expires_at: datetime


# class VerifyCodeRequest(BaseModel):
#     email: EmailStr
#     code: constr(min_length=6, max_length=6)


# class VerifyCodeResponse(BaseModel):
#     token: str
#     user_id: str
#     tenant_id: str
#     email: EmailStr


# @router.post("/auth/request-code", response_model=RequestCodeResponse)
# def request_code(payload: RequestCodeRequest):
#     try:
#         entry = auth_service.request_code(payload.email)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))

#     return {
#         "email": entry.email,
#         "message": "Verification code sent to email",
#         "expires_at": entry.expires_at,
#     }


# @router.post("/auth/verify-code", response_model=VerifyCodeResponse)
# def verify_code(payload: VerifyCodeRequest):
#     try:
#         session = auth_service.verify_code(payload.email, payload.code)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))

#     return {
#         "token": session.token,
#         "user_id": session.user_id,
#         "tenant_id": session.tenant_id,
#         "email": session.email,
#     }


# backend/src/tender_analyzer/apps/api_gateway/auth_routes.py

# backend/src/tender_analyzer/apps/api_gateway/routes/auth_routes.py

from datetime import datetime, timedelta
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from tender_analyzer.common.config.settings import settings
from tender_analyzer.common.auth.jwt import create_access_token
from tender_analyzer.common.auth.utils import is_email_allowed
from tender_analyzer.common.db.session import get_db
from tender_analyzer.domain import models  # <-- we use models directly

router = APIRouter()


# -------------------------------
# Request/response schemas
# -------------------------------

class RequestCodeBody(BaseModel):
    email: EmailStr


class VerifyCodeBody(BaseModel):
    email: EmailStr
    code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    email: EmailStr
    role: str


# -------------------------------
# Helpers
# -------------------------------

def _generate_login_code(length: int = 6) -> str:
    """Generate a numeric one-time code."""
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(length))


def _get_code_expiry() -> datetime:
    """Compute expiry time for a new login code."""
    return datetime.utcnow() + timedelta(minutes=10)


def _get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return (
        db.query(models.User)
        .filter(models.User.email == email)
        .first()
    )


def _get_latest_login_code_for_user(db: Session, user_id) -> Optional[models.LoginCode]:
    return (
        db.query(models.LoginCode)
        .filter(models.LoginCode.user_id == user_id)
        .order_by(models.LoginCode.created_at.desc())
        .first()
    )


# -------------------------------
# /auth/request-code
# -------------------------------

@router.post("/request-code", status_code=status.HTTP_200_OK)
def request_code(body: RequestCodeBody, db: Session = Depends(get_db)):
    """
    Step 1: User submits email.
    We:
      - enforce domain restriction (only @gruner.ch),
      - verify user exists & is active,
      - generate a short-lived one-time code,
      - persist it,
      - send/log it.
    """
    email = body.email.strip().lower()

    # 1) Domain restriction
    if not is_email_allowed(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only @gruner.ch email addresses are allowed to log in.",
        )

    # 2) Look up user
    user = _get_user_by_email(db, email=email)
    if user is None or not getattr(user, "is_active", True):
        # You can return a generic message instead if you prefer
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active user found for this email.",
        )

    # 3) TODO: optional rate limiting (skipped for now)

    # 4) Generate and persist a new login code
    code = _generate_login_code()
    login_code = models.LoginCode(
        user_id=user.id,
        code=code,
        created_at=datetime.utcnow(),
        expires_at=_get_code_expiry(),
        attempt_count=0,
    )
    db.add(login_code)
    db.commit()
    db.refresh(login_code)

    # 5) DEV: log code instead of sending email
    # Replace this with your real email sending in production
    print(f"[auth] verification code for {user.email}: {code}")

    return {"detail": "If this email is registered, a login code has been sent."}


# -------------------------------
# /auth/verify-code
# -------------------------------

@router.post("/verify-code", response_model=TokenResponse)
def verify_code(body: VerifyCodeBody, db: Session = Depends(get_db)):
    """
    Step 2: User submits email + code.
    We:
      - re-enforce the domain restriction,
      - find user,
      - find latest active LoginCode,
      - verify code match, expiry, and attempt limit,
      - generate JWT token with user_id, tenant_id, role,
      - mark code as consumed.
    """
    email = body.email.strip().lower()
    code_input = body.code.strip()

    # 1) Domain restriction again (defense in depth)
    if not is_email_allowed(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not allowed.",
        )

    # 2) Lookup user
    user = _get_user_by_email(db, email=email)
    if user is None or not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or code.",
        )

    # 3) Get latest login code for this user
    login_code = _get_latest_login_code_for_user(db, user_id=user.id)
    if login_code is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or code.",
        )

    now = datetime.utcnow()
    MAX_ATTEMPTS = 5

    # 4) Check attempts, consumed, expired
    if login_code.attempt_count >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This code has been used too many times. Please request a new one.",
        )

    if login_code.consumed_at is not None or login_code.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This code has expired. Please request a new one.",
        )

    # 5) Compare codes
    if login_code.code != code_input:
        login_code.attempt_count += 1
        db.add(login_code)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or code.",
        )

    # 6) Mark code as consumed
    login_code.consumed_at = now
    db.add(login_code)
    db.commit()

    # 7) Generate JWT access token
    token_payload = {
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "email": user.email,
        "role": getattr(user, "role", "user"),
    }
    access_token = create_access_token(token_payload)

    return TokenResponse(
        access_token=access_token,
        user_id=str(user.id),
        tenant_id=str(user.tenant_id),
        email=user.email,
        role=getattr(user, "role", "user"),
    )
