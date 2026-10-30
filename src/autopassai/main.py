"""Application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from autopassai.api.health import healthz_router
from autopassai.api.process import process_router
from autopassai.api.router import api_router
from autopassai.config import settings
from autopassai.logging import configure_logging
from autopassai.services.inference import load_registered_model

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading model: name=%s alias=%s",
        settings.mlflow_model_name,
        settings.mlflow_model_alias,
    )

    loaded = await asyncio.to_thread(
        load_registered_model,
        settings,
    )

    app.state.inference = loaded

    logger.info(
        "Application started: name=%s version=%s model=%s model_version=%s",
        settings.app_name,
        settings.app_version,
        loaded.name,
        loaded.version,
    )

    try:
        yield
    finally:
        del app.state.inference
        logger.info("Application stopped.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(healthz_router)
app.include_router(api_router)
app.include_router(process_router)
