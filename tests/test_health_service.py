"""Tests for health-check services."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from autopassai.services.health import check_database


@pytest.mark.asyncio
async def test_check_database_success() -> None:
    """Test a successful database health check."""
    connection = AsyncMock()
    result = MagicMock()
    result.scalar_one.return_value = "PostgreSQL 18"

    connection.execute.return_value = result

    engine = MagicMock()
    engine.connect.return_value.__aenter__ = AsyncMock(
        return_value=connection,
    )
    engine.connect.return_value.__aexit__ = AsyncMock(
        return_value=None,
    )

    result = await check_database(engine)

    assert result["status"] == "ok"
    assert result["version"] == "PostgreSQL 18"
    assert result["response_time_ms"] >= 0

    connection.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_database_error() -> None:
    """Test a failed database health check."""
    connection = AsyncMock()
    connection.execute.side_effect = RuntimeError("Database unavailable")

    engine = MagicMock()
    engine.connect.return_value.__aenter__ = AsyncMock(
        return_value=connection,
    )
    engine.connect.return_value.__aexit__ = AsyncMock(
        return_value=None,
    )

    result = await check_database(engine)

    assert result["status"] == "error"
    assert result["error"] == "Database unavailable"
    assert result["response_time_ms"] >= 0
