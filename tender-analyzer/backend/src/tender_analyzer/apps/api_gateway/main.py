from fastapi import FastAPI

from tender_analyzer.apps.api_gateway.routes.auth_routes import router as auth_router
from tender_analyzer.apps.api_gateway.routes.tender_routes import router as tender_router
from tender_analyzer.common.config.settings import settings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tender_analyzer.common.db.session import init_db


app = FastAPI(title=settings.app_name)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # IMPORTANT: allow OPTIONS (and POST, etc.)
    allow_headers=["*"],    # IMPORTANT: allow Content-Type, Authorization, etc.
)

@app.on_event("startup")
def on_startup():
    init_db()  # create tables at startup (dev-friendly)

# Now include your routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(tender_router, prefix=settings.api_prefix)
