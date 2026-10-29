import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from autopassai.ml.charset import NUM_CLASSES
from autopassai.ml.dataset import PlateDataset, collate_fn
from autopassai.ml.model import CRNN
from autopassai.ml.preprocessing import build_transform


def main():
    parser = argparse.ArgumentParser(description="Проверка корректности процесса обучения на одном батче.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Папка с lpr.csv и cropped_lps.",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Папка с разбитыми данными.",
    )

    parser.add_argument(
        "--preprocessing",
        type=str,
        required=True,
        choices=["pad", "resize"],
        help="Способ масштабирования изображений.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="",
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Prepare data
    transform = build_transform(
        mode=args.preprocessing,
        height=32,
        width=160,
    )

    dataset = PlateDataset(
        csv_path=args.splits_dir / "train.csv",
        image_dir=args.data_dir / "cropped_lps",
        transform=transform,
    )

    if len(dataset) < 4:
        raise ValueError("Для проверки нужно минимум 4 изображения.")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets, target_lengths, texts = next(iter(loader))
    images = images.to(device)

    # Check data
    if images.shape != (4, 1, 32, 160):
        raise ValueError(f"Неверная форма images: {images.shape}")

    if targets.shape != (24,) or target_lengths.shape != (4,):
        raise ValueError("Неверные формы меток для четырёх номеров длины 6.")

    if target_lengths.sum().item() != targets.numel():
        raise ValueError("Сумма длин меток не совпадает с числом символов.")

    if not torch.all(target_lengths == 6).item():
        raise ValueError("Ожидались номера длиной 6 символов.")

    if targets.min().item() < 1 or targets.max().item() >= NUM_CLASSES:
        raise ValueError("Индексы меток выходят за границы алфавита.")

    # forward
    model = CRNN(
        num_classes=NUM_CLASSES,
        hidden_size=128,
    ).to(device)
    model.train()

    logits = model(images)

    if logits.shape != (40, 4, NUM_CLASSES):
        raise ValueError(f"Неверная форма logits: {logits.shape}")

    # CTC Loss
    log_probs = logits.log_softmax(dim=2)

    input_lengths = torch.full(
        size=(images.size(0),),
        fill_value=logits.size(0),
        dtype=torch.long,
    )

    criterion = nn.CTCLoss(blank=0)

    loss = criterion(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
    )

    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Некорректный loss: {loss.item()}")

    # backward
    loss.backward()

    has_nonzero_gradient = False

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        gradient = parameter.grad

        if gradient is None:
            raise RuntimeError(f"Нет градиента у параметра {name}")

        if not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"Некорректный градиент у параметра {name}")

        if torch.count_nonzero(gradient).item() > 0:
            has_nonzero_gradient = True

    if not has_nonzero_gradient:
        raise RuntimeError("Все градиенты равны нулю.")

    parameter_count = sum(p.numel() for p in model.parameters())

    if parameter_count != 1_319_459:
        raise RuntimeError(f"Изменилось число параметров: {parameter_count}")

    # Save preprocessed image
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preview_tensor = (images[0].detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    preview_path = args.output_dir / f"sample_{args.preprocessing}.png"
    to_pil_image(preview_tensor).save(preview_path)

    print("Preprocessing:", args.preprocessing)
    print("images:", images.shape)
    print("targets:", targets.shape)
    print("target_lengths:", target_lengths.shape)
    print("logits:", logits.shape)
    print("Число параметров:", parameter_count)
    print("CTC Loss:", loss.item())
    print("Правильные ответы:", texts)
    print("Градиенты конечны, есть ненулевые значения.")
    print("Пример сохранён:", preview_path)
    print("Проверка пройдена.")


if __name__ == "__main__":
    main()
