import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
from PIL import Image

from autopassai.ml.dataset_tracking import log_dataset_inputs

matplotlib.use("Agg")


def read_rows(csv_path):
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = [{"images": row["images"], "labels": row["labels"]} for row in csv.DictReader(file)]

    if not rows:
        raise ValueError("CSV-файл не содержит примеров.")

    return rows


def build_summary(output_dir, train_rows, val_rows, test_rows):
    result = {}

    splits = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }

    for split_part, rows in splits.items():
        label_counts = Counter(row["labels"] for row in rows)
        char_counts = Counter(char for row in rows for char in row["labels"])

        result[split_part] = {
            "size": len(rows),
            "labels_sizes": dict(Counter(len(row["labels"]) for row in rows)),
            "chars_frequencies": dict(char_counts),
            "unique_labels": len(label_counts),
            "repeated_labels": sum(count > 1 for count in label_counts.values()),
            "extra_examples": sum(count - 1 for count in label_counts.values()),
            "alphabet": "".join(sorted(set("".join(row["labels"] for row in rows)))),
        }

    output_path = output_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result


def save_image_sizes(train_rows, image_dir, output_dir):
    widths = []
    heights = []
    aspect_ratios = []

    for row in train_rows:
        image_path = image_dir / row["images"]

        with Image.open(image_path) as image:
            width, height = image.size

        widths.append(width)
        heights.append(height)
        aspect_ratios.append(width / height)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    distributions = [
        (widths, "Image width", "Width, pixels"),
        (heights, "Image height", "Height, pixels"),
        (aspect_ratios, "Aspect ratio", "Width / Height"),
    ]

    for ax, (values, title, xlabel) in zip(axes, distributions, strict=True):
        ax.hist(values, bins=40, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of images")

    fig.tight_layout()

    output_path = output_dir / "image_sizes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_char_frequency(train_rows, output_dir):
    frequencies = Counter(char for row in train_rows for char in row["labels"])

    characters = sorted(frequencies)
    counts = [frequencies[char] for char in characters]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(characters, counts)
    ax.set_title("Character frequency in train")
    ax.set_xlabel("Character")
    ax.set_ylabel("Number of occurrences")

    fig.tight_layout()

    output_path = output_dir / "character_frequency.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_train_samples(train_rows, image_dir, output_dir, seed=42):
    selected = random.Random(seed).sample(
        train_rows,
        k=min(24, len(train_rows)),
    )

    fig, axes = plt.subplots(6, 4, figsize=(16, 18))

    for ax in axes.flat:
        ax.axis("off")

    for ax, row in zip(axes.flat, selected, strict=False):
        image_path = image_dir / row["images"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            ax.imshow(image)

        ax.set_title(
            f"{row['images']}\nLabel: {row['labels']}",
            fontsize=10,
        )

    fig.tight_layout()

    output_path = output_dir / "train_samples.jpg"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="")

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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed.",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5050",
        help="Адрес mlflow-сервера.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="autopassai-eda",
        help="Название эксперимента.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="eda-original",
        help="Имя Run в mlflow.",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Папка с manifest и dataset_metadata.json.",
    )

    args = parser.parse_args()

    image_dir = args.data_dir / "cropped_lps"
    output_dir = args.output_dir
    splits_dir = args.splits_dir
    conclusions_path = output_dir / "conclusions.md"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {image_dir}")

    if not conclusions_path.is_file():
        raise FileNotFoundError(f"Нет файла выводов: {conclusions_path}")

    train_rows = read_rows(splits_dir / "train.csv")
    val_rows = read_rows(splits_dir / "val.csv")
    test_rows = read_rows(splits_dir / "test.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tag("stage", "eda")

        log_dataset_inputs(
            manifest_dir=args.manifest_dir,
            splits=["train", "val", "test"],
            splits_dir=splits_dir,
            context="eda",
        )

        mlflow.log_params(
            {
                "seed": args.seed,
                "sample_count": min(24, len(train_rows)),
                "data_dir": str(args.data_dir.resolve()),
                "splits_dir": str(args.splits_dir.resolve()),
            }
        )

        summary = build_summary(
            output_dir,
            train_rows,
            val_rows,
            test_rows,
        )

        save_image_sizes(train_rows, image_dir, output_dir)
        save_char_frequency(train_rows, output_dir)
        save_train_samples(train_rows, image_dir, output_dir, args.seed)

        metrics = {}

        for name, stats in summary.items():
            metrics[f"{name}_size"] = stats["size"]
            metrics[f"{name}_unique_labels"] = stats["unique_labels"]
            metrics[f"{name}_repeated_labels"] = stats["repeated_labels"]
            metrics[f"{name}_extra_examples"] = stats["extra_examples"]

        metrics["train_alphabet_size"] = len(summary["train"]["alphabet"])

        mlflow.log_metrics(metrics)

        artifact_names = [
            "summary.json",
            "image_sizes.png",
            "character_frequency.png",
            "train_samples.jpg",
            "conclusions.md",
        ]

        for filename in artifact_names:
            mlflow.log_artifact(
                str(output_dir / filename),
                artifact_path="eda",
            )

        run_id = run.info.run_id

    print("EDA сохранён в MLflow.")
    print("Run ID:", run_id)


if __name__ == "__main__":
    main()
