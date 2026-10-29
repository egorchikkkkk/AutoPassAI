import base64
import binascii
from io import BytesIO

import pandas as pd
import torch
from mlflow.pyfunc import PythonModel
from PIL import Image

from autopassai.ml.charset import ALPHABET, NUM_CLASSES
from autopassai.ml.decoding import decode_predictions
from autopassai.ml.model import CRNN
from autopassai.ml.preprocessing import build_transform


class PlateRecognitionModel(PythonModel):
    def load_context(self, context):
        checkpoint = torch.load(
            context.artifacts["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["alphabet"] != ALPHABET:
            raise ValueError("Alphabet of checkpoint is not equal to current alphabet.")

        config = checkpoint["config"]

        self.model = CRNN(
            num_classes=NUM_CLASSES,
            hidden_size=config["model"]["hidden_size"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        preprocessing = config["preprocessing"]

        self.transform = build_transform(
            mode=preprocessing["mode"],
            height=preprocessing["height"],
            width=preprocessing["width"],
        )

    def predict(self, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Pandas DataFrame is needed.")

        if not model_input.columns.is_unique:
            raise ValueError("Column names should be unique.")

        if "image_base64" not in model_input.columns:
            raise ValueError("There is no 'image_base64' column.")

        if model_input.empty:
            return pd.DataFrame({"prediction": pd.Series(index=model_input.index, dtype=str)})

        tensors = []

        for position, (index, encoded) in enumerate(model_input["image_base64"].items()):
            if not isinstance(encoded, str):
                raise ValueError(f"Row {position}, index {index!r}: image_base64 should be 'str'.")

            try:
                image_bytes = base64.b64decode(
                    encoded,
                    validate=True,
                )

                with Image.open(BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    tensor = self.transform(image)

            except (binascii.Error, OSError, ValueError) as error:
                raise ValueError(f"Row {position}, index {index!r}: invalid base64 or image-file.") from error

            tensors.append(tensor)

        images = torch.stack(tensors)

        with torch.inference_mode():
            logits = self.model(images)
            predictions = decode_predictions(logits)

        return pd.DataFrame(
            {"prediction": predictions},
            index=model_input.index,
        )
