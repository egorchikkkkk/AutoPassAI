"""Main API router configuration."""

from fastapi import APIRouter

from autopassai.api.health import api_health_router
from autopassai.api.version import router as version_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(api_health_router)
api_router.include_router(version_router)
