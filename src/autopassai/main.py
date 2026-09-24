"""Application entry point."""

from fastapi import FastAPI

from autopassai.api.health import healthz_router
from autopassai.api.router import api_router
from autopassai.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(healthz_router)
app.include_router(api_router)
