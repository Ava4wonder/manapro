import uuid


def generate_id(prefix: str = "tender") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
