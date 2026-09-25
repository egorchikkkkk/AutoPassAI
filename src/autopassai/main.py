"""Application entry point."""

import logging

from fastapi import FastAPI

from autopassai.api.health import healthz_router
from autopassai.api.router import api_router
from autopassai.config import settings
from autopassai.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(healthz_router)
app.include_router(api_router)

logger.info(
    "Application started: name=%s version=%s",
    settings.app_name,
    settings.app_version,
)
