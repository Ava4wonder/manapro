from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tender_analyzer.common.auth.models import AuthenticatedUser
from tender_analyzer.common.auth.service import auth_service

security = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Missing or invalid authorization token") -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    if not credentials:
        raise _unauthorized()

    session = auth_service.get_session(credentials.credentials)
    if not session:
        raise _unauthorized("Session is missing or expired")

    return AuthenticatedUser(
        user_id=session.user_id,
        tenant_id=session.tenant_id,
        email=session.email,
        token=session.token,
    )
