import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from autopassai.ml.charset import CHAR_TO_IDX


class PlateDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        with Path(csv_path).open(encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image_path = self.image_dir / row["images"]
        text = row["labels"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(
            [CHAR_TO_IDX[char] for char in text],
            dtype=torch.long,
        )

        return image, target, text


def collate_fn(batch):
    images, targets, texts = zip(*batch, strict=True)

    images = torch.stack(images)

    target_lengths = torch.tensor(
        [len(target) for target in targets],
        dtype=torch.long,
    )

    targets = torch.cat(targets)

    return images, targets, target_lengths, list(texts)
