# backend/src/tender_analyzer/common/auth/utils.py

from ..config.settings import get_allowed_domains


def is_email_allowed(email: str) -> bool:
    """
    Check if the email's domain is allowed to log in.

    Uses ALLOWED_LOGIN_EMAIL_DOMAINS from settings, for example:
        ALLOWED_LOGIN_EMAIL_DOMAINS=gruner.ch,example.com
    """
    email = email.strip().lower()
    if "@" not in email:
        return False

    local_part, domain = email.split("@", 1)
    if not domain:
        return False

    allowed = set(get_allowed_domains())  # e.g. {"gruner.ch"}
    return domain in allowed
