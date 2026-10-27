import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Evaluate file SHA-256 (chunk reading)."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_manifest(csv_path: Path, image_dir: Path, output_path: Path) -> dict:
    """Create sample manifest and return its metadata."""
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"images", "labels"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"В {csv_path} нужны столбцы images и labels.")

        rows = list(reader)

    if not rows:
        raise ValueError(f"Пустая выборка: {csv_path}")

    rows.sort(key=lambda row: row["images"])

    manifest_rows = []

    for row in rows:
        filename = row["images"]
        label = row["labels"]

        if not filename or not label:
            raise ValueError(f"Пустое имя файла или метка в {csv_path}")

        image_path = image_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(f"Нет изображения: {image_path}")

        manifest_rows.append(
            {
                "images": filename,
                "labels": label,
                "sha256": file_sha256(image_path),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["images", "labels", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "rows": len(manifest_rows),
        "manifest_file": output_path.name,
        "manifest_sha256": file_sha256(output_path),
        "split_csv_sha256": file_sha256(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Создание manifest и контрольных сумм датасета.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Папка с исходными данными.",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Папка с разбитыми данными.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Папка для сохранения результатов EDA.",
    )

    args = parser.parse_args()

    image_dir = args.data_dir / "cropped_lps"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": args.data_dir.resolve().as_uri(),
        "hash_algorithm": "sha256",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        csv_path = args.splits_dir / f"{split}.csv"
        output_path = args.output_dir / f"{split}_manifest.csv"

        split_metadata = build_manifest(
            csv_path=csv_path,
            image_dir=image_dir,
            output_path=output_path,
        )

        metadata["splits"][split] = split_metadata

        print(f"{split}: {split_metadata['rows']} примеров")
        print("SHA-256:", split_metadata["manifest_sha256"])

    metadata_path = args.output_dir / "dataset_metadata.json"

    with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("Метаданные сохранены:", metadata_path)


if __name__ == "__main__":
    main()
