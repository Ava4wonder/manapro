# backend/src/tender_analyzer/common/auth/jwt.py

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import jwt, JWTError

from ..config.settings import settings


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    `data` is your payload, e.g.:
        {
          "sub": user_id,
          "tenant_id": tenant_id,
          "email": email,
          "role": role,
        }

    We add standard `exp` and `iat` claims.
    """
    to_encode = data.copy()

    now = datetime.utcnow()
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_EXPIRES_MINUTES)

    expire = now + expires_delta

    # Standard JWT claims
    to_encode.update({
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode & verify a JWT access token.

    Raises JWTError if invalid/expired.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as exc:
        # You can wrap this in your own exception type if you like.
        raise exc
