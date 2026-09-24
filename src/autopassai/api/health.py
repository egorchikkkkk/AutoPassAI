"""Health-check API endpoints."""

from fastapi import APIRouter

from autopassai.db.session import engine
from autopassai.services.health import check_database

healthz_router = APIRouter()
api_health_router = APIRouter()


@healthz_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a quick response indicating that the application is running."""
    return {"status": "ok"}


@api_health_router.get("/health")
async def health() -> dict[str, object]:
    """Return the health status of application dependencies."""
    database = await check_database(engine)

    status = "ok" if database["status"] == "ok" else "error"

    return {
        "status": status,
        "components": {
            "database": database,
        },
    }
