from uuid import uuid4

from tender_analyzer.common.db.session import SessionLocal, init_db
from tender_analyzer.domain.models import User

def main():
    # ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        # Check if user already exists
        email = "jiahui.ye@gruner.ch"
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"User {email} already exists with id={existing.id}")
            return

        # For now, just create a dummy tenant_id
        tenant_id = uuid4()

        user = User(
            id=uuid4(),
            tenant_id=tenant_id,
            email=email,
            role="admin",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created user {email} with id={user.id}, tenant_id={user.tenant_id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
