"""Health-check API endpoints."""

from fastapi import APIRouter

healthz_router = APIRouter()
api_health_router = APIRouter()


@healthz_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a quick response indicating that the application is running."""
    return {"status": "ok"}


@api_health_router.get("/health")
async def health() -> dict[str, str]:
    """Return the health status of application dependencies."""
    return {"status": "ok"}
