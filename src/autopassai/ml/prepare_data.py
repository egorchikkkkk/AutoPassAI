import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Проверка датасета и создание train/val/test.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Папка с lpr.csv и cropped_lps.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Папка для сохранения разбиений.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed перемешивания групп.",
    )

    args = parser.parse_args()

    image_dir = args.data_dir / "cropped_lps"
    csv_path = args.data_dir / "lpr.csv"
    output_dir = args.output_dir

    # Read file names and labels
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = [{"images": row["images"], "labels": row["labels"]} for row in csv.DictReader(file)]

    if not rows:
        raise ValueError("CSV-файл не содержит примеров.")

    print("Всего примеров:", len(rows))
    print("Длины номеров:", dict(Counter(len(row["labels"]) for row in rows)))

    alphabet = "".join(sorted(set("".join(row["labels"] for row in rows))))
    print("Алфавит:", alphabet)
    print("Количество символов:", len(alphabet))

    # Check images readability
    problems = []

    for row in rows:
        path = image_dir / row["images"]

        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError) as error:
            problems.append((row["images"], str(error)))

    if problems:
        for filename, error in problems[:10]:
            print(f"Проблема с {filename}: {error}")

        raise RuntimeError(
            f"Найдено проблемных изображений: {len(problems)}. Сначала нужно разобраться с ними."
        )

    print("Все изображения успешно прочитаны.")

    # Join images with same labels
    groups = defaultdict(list)

    for row in rows:
        groups[row["labels"]].append(row)

    labels = sorted(groups)
    random.Random(args.seed).shuffle(labels)

    # Split data: train/val/test
    train_end = int(len(labels) * 0.8)
    val_end = train_end + int(len(labels) * 0.1)

    split_labels = {
        "train": labels[:train_end],
        "val": labels[train_end:val_end],
        "test": labels[val_end:],
    }

    splits = {
        name: [row for label in selected_labels for row in groups[label]]
        for name, selected_labels in split_labels.items()
    }

    # Check labels disjoint between splits (to avoid dataleak)
    label_sets = {name: set(selected_labels) for name, selected_labels in split_labels.items()}

    assert label_sets["train"].isdisjoint(label_sets["val"])
    assert label_sets["train"].isdisjoint(label_sets["test"])
    assert label_sets["val"].isdisjoint(label_sets["test"])
    assert sum(len(part) for part in splits.values()) == len(rows)

    # Check that val and test labels chars are contained in train labels
    # (otherwise model won't be able to recognize them)
    train_chars = set("".join(row["labels"] for row in splits["train"]))

    for name in ("val", "test"):
        chars = set("".join(row["labels"] for row in splits[name]))
        unseen = chars - train_chars

        if unseen:
            raise ValueError(f"В {name} есть незнакомые train символы: {unseen}")

    # Save only names (no need to copy images)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        output_path = output_dir / f"{name}.csv"

        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["images", "labels"])
            writer.writeheader()
            writer.writerows(part)

        print(f"{name}: {len(part)} изображений → {output_path}")

    print("Разделение завершено. Пересечений по текстам номеров нет.")


if __name__ == "__main__":
    main()
