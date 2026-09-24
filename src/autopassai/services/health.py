"""Services for checking application dependencies."""

from time import perf_counter

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


async def check_database(engine: AsyncEngine) -> dict[str, object]:
    """Check PostgreSQL availability, version, and response time."""
    started_at = perf_counter()

    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT version()"))
            database_version = result.scalar_one()

        response_time_ms = (perf_counter() - started_at) * 1000

        return {"status": "ok", "version": database_version, "response_time_ms": round(response_time_ms, 2)}

    except Exception as exc:
        response_time_ms = (perf_counter() - started_at) * 1000

        return {"status": "error", "error": str(exc), "response_time_ms": round(response_time_ms, 2)}
