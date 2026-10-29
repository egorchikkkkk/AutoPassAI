import torch
from torch import nn

from autopassai.ml.decoding import decode_predictions
from autopassai.ml.metrics import evaluate_metrics


def calculate_loss(logits, targets, target_lengths, criterion):
    log_probs = logits.float().log_softmax(dim=2).cpu()

    input_lengths = torch.full(
        size=(logits.size(1),),
        fill_value=logits.size(0),
        dtype=torch.long,
    )

    return criterion(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, gradient_limit=5):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for step, (images, targets, target_lengths, _) in enumerate(loader, start=1):
        images = images.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = calculate_loss(logits, targets, target_lengths, criterion)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Invalid loss on batch {step}: {loss.item()}")

        loss.backward()

        # Limit gradient norm for training stability
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_limit,
            error_if_nonfinite=True,
        )

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if step % 100 == 0 or step == len(loader):
            print(
                f"  Батч {step}/{len(loader)} | train loss: {total_loss / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("Train sample is empty.")

    return total_loss / total_samples


# Validation for model quality evaluation and training stopping (if quality doesn't raise)
@torch.no_grad()
def validate(model, loader, criterion, device):
    # switch model to prediction mode (switch off dropout and etc.)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_references = []
    all_predictions = []

    for step, (images, targets, target_lengths, texts) in enumerate(loader, start=1):
        images = images.to(device)

        logits = model(images)
        loss = calculate_loss(logits, targets, target_lengths, criterion)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Invalid validation loss on batch {step}: {loss.item()}.")

        predictions = decode_predictions(logits)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_references.extend(texts)
        all_predictions.extend(predictions)

    if total_samples == 0:
        raise ValueError("Validation-sample is empty.")

    accuracy, cer = evaluate_metrics(all_references, all_predictions)

    prediction_rows = [
        {
            "target": reference,
            "prediction": prediction,
        }
        for reference, prediction in zip(all_references, all_predictions, strict=True)
    ]

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "cer": cer,
        "predictions": prediction_rows,
    }
