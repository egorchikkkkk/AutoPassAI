"""Tests for health-check API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from autopassai.main import app


@pytest.mark.asyncio
async def test_healthz() -> None:
    """Test the lightweight application health endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_version() -> None:
    """Test the application version endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/version")

    assert response.status_code == 200
    assert "version" in response.json()
    assert response.json()["version"] == "0.1.2"


@pytest.mark.asyncio
async def test_health() -> None:
    """Test the dependency health endpoint."""
    mock_database = {
        "status": "ok",
        "version": "PostgreSQL 18",
        "response_time_ms": 1.23,
    }

    with patch(
        "autopassai.api.health.check_database",
        new=AsyncMock(return_value=mock_database),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["components"]["database"]["status"] == "ok"
    assert data["components"]["database"]["version"] == "PostgreSQL 18"
    assert data["components"]["database"]["response_time_ms"] >= 0


@pytest.mark.asyncio
async def test_health_when_database_is_unavailable() -> None:
    """Test the dependency health endpoint when the database is unavailable."""
    mock_database = {
        "status": "error",
        "error": "Database connection failed",
        "response_time_ms": 2.5,
    }

    with patch(
        "autopassai.api.health.check_database",
        new=AsyncMock(return_value=mock_database),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "error"
    assert data["components"]["database"]["status"] == "error"
    assert data["components"]["database"]["response_time_ms"] >= 0
