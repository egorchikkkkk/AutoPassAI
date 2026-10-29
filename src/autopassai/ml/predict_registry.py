import argparse
import base64
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser(description="Plate recognition by model from mlflow Registry.")

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="autopassai-crnn")
    parser.add_argument("--alias", type=str, default="champion")
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5050")

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image is not found: {args.image}")

    # Set server address for getting model and its artifacts
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)

    client = MlflowClient()

    # Get model version by alias
    model_version = client.get_model_version_by_alias(
        name=args.model_name,
        alias=args.alias,
    )

    # Get model
    model_uri = f"models:/{args.model_name}/{model_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Format image according to input format of PyFunc-wrap
    encoded_image = base64.b64encode(args.image.read_bytes()).decode("ascii")

    model_input = pd.DataFrame(
        {
            "image_base64": [encoded_image],
        }
    )

    result = model.predict(model_input)
    prediction = result["prediction"].iloc[0]

    print("Model:", args.model_name)
    print("Alias:", args.alias)
    print("Loaded version:", model_version.version)
    print("Origin Run ID:", model_version.run_id)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
