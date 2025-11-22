from pydantic import BaseModel, EmailStr


class AuthenticatedUser(BaseModel):
    user_id: str
    tenant_id: str
    email: EmailStr
    token: str
