import argparse
import base64
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import mlflow
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from PIL import Image

from autopassai.ml.charset import ALPHABET, NUM_CLASSES
from autopassai.ml.decoding import decode_predictions
from autopassai.ml.model import CRNN
from autopassai.ml.preprocessing import build_transform
from autopassai.ml.pyfunc_model import PlateRecognitionModel


def main():
    parser = argparse.ArgumentParser(description="Register model.")

    parser.add_argument("--run-id", type=str, required=True, help="ID of a run with needed checkpoint.")

    parser.add_argument("--example-image", type=Path, required=True, help="Image for check.")

    parser.add_argument("--model-name", type=str, default="autopassai-crnn", help="Model name.")

    parser.add_argument(
        "--tracking-uri", type=str, default="http://127.0.0.1:5050", help="Mlflow-server address."
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    checkpoint_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{args.run_id}/checkpoints/best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["run_id"] != args.run_id:
        raise ValueError(
            f"Checkpoint belongs to run {checkpoint['run_id']}, but requested run is {args.run_id}."
        )

    if checkpoint["alphabet"] != ALPHABET:
        raise ValueError("Checkpoint alphabet does not match the current alphabet.")

    config = checkpoint["config"]

    model = CRNN(
        num_classes=NUM_CLASSES,
        hidden_size=config["model"]["hidden_size"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(
        mode=config["preprocessing"]["mode"],
        height=config["preprocessing"]["height"],
        width=config["preprocessing"]["width"],
    )

    image_bytes = args.example_image.read_bytes()

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        tensor = transform(image)

    # Add batch dimension: [1, 32, 160] → [1, 1, 32, 160].
    images = tensor.unsqueeze(0)

    with torch.inference_mode():
        logits = model(images)
        reference_predictions = decode_predictions(logits)

    print("Reference prediction:", reference_predictions[0])

    encoded_image = base64.b64encode(image_bytes).decode("ascii")

    input_example = pd.DataFrame(
        {
            "image_base64": [encoded_image],
        }
    )

    signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", "image_base64"),
            ]
        ),
        outputs=Schema(
            [
                ColSpec("string", "prediction"),
            ]
        ),
    )

    dependencies = [
        "mlflow",
        "torch",
        "torchvision",
        "pillow",
        "pandas",
        "cloudpickle",
    ]

    requirements = [f"{package}=={version(package)}" for package in dependencies]

    package_dir = Path(__file__).resolve().parents[1]

    with mlflow.start_run(run_id=args.run_id):
        model_info = mlflow.pyfunc.log_model(
            name="ocr",
            python_model=PlateRecognitionModel(),
            artifacts={
                "checkpoint": checkpoint_path,
            },
            code_paths=[str(package_dir)],
            signature=signature,
            input_example=input_example,
            pip_requirements=requirements,
        )

        loaded_model = mlflow.pyfunc.load_model(
            model_info.model_uri,
        )

        result = loaded_model.predict(input_example)
        packaged_predictions = result["prediction"].tolist()

        if packaged_predictions != reference_predictions:
            raise RuntimeError(
                "Predictions changed after packaging: "
                f"reference={reference_predictions}, "
                f"packaged={packaged_predictions}"
            )

        print("Packaging check passed.")

    registered_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=args.model_name,
    )

    print("Source Run ID:", args.run_id)
    print("Registered model:", registered_version.name)
    print("Model version:", registered_version.version)
    print("Model URI:", model_info.model_uri)
    print("Prediction:", packaged_predictions[0])


if __name__ == "__main__":
    main()
