"""Version API endpoint."""

from fastapi import APIRouter

from autopassai.config import settings

router = APIRouter()


@router.get("/version")
async def get_version() -> dict[str, str]:
    """Return the current application version."""
    return {"version": settings.app_version}
