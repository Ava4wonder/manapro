from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import random

from tender_analyzer.common.auth.permissions import ALLOWED_EMAIL_DOMAIN
from tender_analyzer.common.utils.ids import generate_id


@dataclass
class AuthCodeEntry:
    email: str
    code: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class AuthSession:
    token: str
    user_id: str
    tenant_id: str
    email: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class AuthService:
    _CODE_TTL = timedelta(minutes=10)
    _SESSION_TTL = timedelta(hours=4)
    _CODE_LENGTH = 6

    def __init__(self) -> None:
        self._codes: Dict[str, AuthCodeEntry] = {}
        self._sessions: Dict[str, AuthSession] = {}

    def request_code(self, email: str) -> AuthCodeEntry:
        normalized = email.strip().lower()
        if not normalized.endswith(ALLOWED_EMAIL_DOMAIN):
            raise ValueError(f"Only {ALLOWED_EMAIL_DOMAIN} emails are allowed")

        code = str(random.randint(0, 10 ** self._CODE_LENGTH - 1)).zfill(self._CODE_LENGTH)
        expires_at = datetime.now(timezone.utc) + self._CODE_TTL
        entry = AuthCodeEntry(email=normalized, code=code, expires_at=expires_at)
        self._codes[normalized] = entry

        # Simulate sending an email in this prototype by logging the code.
        print(f"[auth] verification code for {normalized}: {code}")

        return entry

    def verify_code(self, email: str, code: str) -> AuthSession:
        normalized = email.strip().lower()
        entry = self._codes.get(normalized)
        if not entry or entry.is_expired() or entry.code != code:
            raise ValueError("Invalid or expired verification code")

        # Invalidate code immediately.
        del self._codes[normalized]

        token = generate_id("token")
        session = AuthSession(
            token=token,
            user_id=normalized,
            tenant_id=normalized.split("@", 1)[-1],
            email=normalized,
            expires_at=datetime.now(timezone.utc) + self._SESSION_TTL,
        )
        self._sessions[token] = session
        return session

    def get_session(self, token: str) -> Optional[AuthSession]:
        session = self._sessions.get(token)
        if not session:
            return None

        if session.is_expired():
            self._sessions.pop(token, None)
            return None

        return session


auth_service = AuthService()
