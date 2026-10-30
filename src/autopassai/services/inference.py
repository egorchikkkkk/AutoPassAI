"""Loading registered model for application inference."""

import base64
from dataclasses import dataclass
from io import BytesIO

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel
from PIL import Image

from autopassai.config import Settings


@dataclass(frozen=True)
class LoadedModel:
    model: PyFuncModel
    name: str
    alias: str
    version: str
    run_id: str | None


def load_registered_model(settings: Settings) -> LoadedModel:
    """Resolve the alias and load that exact model version."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )

    model_uri = f"models:/{settings.mlflow_model_name}/{model_version.version}"

    model = mlflow.pyfunc.load_model(model_uri)

    return LoadedModel(
        model=model,
        name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
        version=model_version.version,
        run_id=model_version.run_id,
    )


class InvalidImageError(ValueError):
    """The uploaded file is not a readable image."""


def predict_image(loaded: LoadedModel, image_bytes: bytes) -> str:
    """Recognize one image using an already loaded model."""
    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
    except (OSError, ValueError) as error:
        raise InvalidImageError("Cannot read the uploaded image.") from error

    encoded = base64.b64encode(image_bytes).decode("ascii")

    model_input = pd.DataFrame(
        {
            "image_base64": [encoded],
        }
    )

    result = loaded.model.predict(model_input)

    return result["prediction"].iloc[0]
