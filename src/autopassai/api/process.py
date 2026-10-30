"""Image recognition endpoint."""

import asyncio

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from autopassai.services.inference import (
    InvalidImageError,
    predict_image,
)

process_router = APIRouter(tags=["inference"])


class ProcessResponse(BaseModel):
    prediction: str
    model_name: str
    model_version: str
    model_alias: str
    run_id: str | None


@process_router.post("/process", response_model=ProcessResponse)
async def process(
    request: Request,
    file: UploadFile,
) -> ProcessResponse:
    loaded = getattr(request.app.state, "inference", None)

    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not loaded.",
        )

    try:
        image_bytes = await file.read()
    finally:
        await file.close()

    try:
        prediction = await asyncio.to_thread(
            predict_image,
            loaded,
            image_bytes,
        )
    except InvalidImageError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return ProcessResponse(
        prediction=prediction,
        model_name=loaded.name,
        model_version=loaded.version,
        model_alias=loaded.alias,
        run_id=loaded.run_id,
    )
