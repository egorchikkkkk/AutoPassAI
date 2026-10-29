import csv
import json
from pathlib import Path

import mlflow
import pandas as pd

from autopassai.ml.data_manifest import file_sha256


def verify_image_contents(manifest_dir: Path, image_dir: Path, splits: list[str]) -> dict[str, str]:
    with (manifest_dir / "dataset_metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    manifest_hashes = {}

    for split in splits:
        info = metadata["splits"][split]
        manifest_path = manifest_dir / info["manifest_file"]

        actual_hash = file_sha256(manifest_path)

        if actual_hash != info["manifest_sha256"]:
            raise ValueError(f"Manifest изменился: {manifest_path}")

        with manifest_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        for row in rows:
            image_path = image_dir / row["images"]

            if file_sha256(image_path) != row["sha256"]:
                raise ValueError(f"Изображение изменилось: {image_path}")

        manifest_hashes[split] = actual_hash

        print(f"{split}: проверено {len(rows)} изображений", flush=True)

    return manifest_hashes


def log_dataset_inputs(manifest_dir: Path, splits: list[str], splits_dir: Path, context: str) -> None:
    if mlflow.active_run() is None:
        raise RuntimeError("Сначала нужно открыть MLflow Run.")

    metadata_path = manifest_dir / "dataset_metadata.json"

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    # Check data correctness
    prepared = []

    for split in splits:
        if split not in metadata["splits"]:
            raise ValueError(f"Неизвестная выборка: {split}")

        split_metadata = metadata["splits"][split]
        manifest_path = manifest_dir / split_metadata["manifest_file"]

        split_path = splits_dir / f"{split}.csv"

        if file_sha256(split_path) != split_metadata["split_csv_sha256"]:
            raise ValueError(f"Разбиение изменилось: {split_path}. Нужно пересоздать manifest.")

        actual_hash = file_sha256(manifest_path)
        expected_hash = split_metadata["manifest_sha256"]

        if actual_hash != expected_hash:
            raise ValueError(f"Контрольная сумма {manifest_path} не совпадает с dataset_metadata.json.")

        frame = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        expected_columns = ["images", "labels", "sha256"]

        if list(frame.columns) != expected_columns:
            raise ValueError(f"Неверные столбцы в {manifest_path}: {list(frame.columns)}")

        if len(frame) != split_metadata["rows"]:
            raise ValueError(
                f"Неверное количество строк в {manifest_path}: {len(frame)} вместо {split_metadata['rows']}"
            )

        if frame.empty or frame.eq("").any().any():
            raise ValueError(f"Пустая таблица или значения в {manifest_path}")

        dataset = mlflow.data.from_pandas(
            df=frame,
            name=f"autopassai-{split}",
            targets="labels",
            source=manifest_path.resolve().as_uri(),
            digest=actual_hash[:32],
        )

        prepared.append((split, dataset, manifest_path, actual_hash))

    # After successful check link data with current Run
    for split, dataset, manifest_path, full_hash in prepared:
        mlflow.log_input(
            dataset,
            context=context,
            tags={
                "split": split,
                "manifest_sha256": full_hash,
                "raw_data_source": metadata["source"],
            },
        )

        mlflow.log_artifact(
            str(manifest_path),
            artifact_path="dataset",
        )

    mlflow.log_artifact(
        str(metadata_path),
        artifact_path="dataset",
    )
