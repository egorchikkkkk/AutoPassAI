import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from autopassai.ml.charset import ALPHABET, NUM_CLASSES
from autopassai.ml.dataset import PlateDataset, collate_fn
from autopassai.ml.dataset_tracking import log_dataset_inputs, verify_image_contents
from autopassai.ml.experiment_config import load_config
from autopassai.ml.model import CRNN
from autopassai.ml.preprocessing import build_transform
from autopassai.ml.training import train_one_epoch, validate


def main():
    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument("--config", type=Path, required=True, help="Experiment configuration TOML-file.")

    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory with lpr.csv and cropped_lps."
    )

    parser.add_argument("--splits-dir", type=Path, required=True, help="Directory with train/val/test.")

    parser.add_argument(
        "--manifest-dir", type=Path, required=True, help="Directory with manifest and metadata."
    )

    parser.add_argument("--output-dir", type=Path, required=True, help="Target saving directory.")

    parser.add_argument(
        "--tracking-uri", type=str, default="http://127.0.0.1:5050", help="Mlflow-server address."
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for training and validation.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable on current device. "
            f"cpu: {torch.cpu.is_available()}, "
            f"cuda: {torch.cuda.is_available()}, "
            f"mps: {torch.backends.mps.is_available()}"
        )
    elif args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS is unavailable on current device. "
            f"cpu: {torch.cpu.is_available()}, "
            f"cuda: {torch.cuda.is_available()}, "
            f"mps: {torch.backends.mps.is_available()}"
        )

    if args.device != "auto":
        device = torch.device(args.device)

    config = load_config(args.config)

    training_config = config["training"]
    preprocessing_config = config["preprocessing"]
    model_config = config["model"]

    seed = training_config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    transform = build_transform(
        mode=preprocessing_config["mode"],
        height=preprocessing_config["height"],
        width=preprocessing_config["width"],
    )

    image_dir = args.data_dir / "cropped_lps"

    train_dataset = PlateDataset(
        csv_path=args.splits_dir / "train.csv", image_dir=image_dir, transform=transform
    )

    val_dataset = PlateDataset(csv_path=args.splits_dir / "val.csv", image_dir=image_dir, transform=transform)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation datasets must not be empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )

    model = CRNN(num_classes=NUM_CLASSES, hidden_size=model_config["hidden_size"]).to(device)

    criterion = nn.CTCLoss(blank=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["experiment"]["run_name"]) as run:
        run_id = run.info.run_id

        run_dir = args.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        print("Run ID:", run_id, flush=True)
        print("Results:", run_dir, flush=True)

        mlflow.set_tag("stage", "training")

        manifest_hashes = verify_image_contents(
            manifest_dir=args.manifest_dir, image_dir=image_dir, splits=["train", "val"]
        )

        log_dataset_inputs(
            manifest_dir=args.manifest_dir, splits=["train"], splits_dir=args.splits_dir, context="training"
        )

        log_dataset_inputs(
            manifest_dir=args.manifest_dir, splits=["val"], splits_dir=args.splits_dir, context="validation"
        )

        parameters = {
            f"{section}.{name}": value for section, values in config.items() for name, value in values.items()
        }

        parameters.update(
            {
                "device": str(device),
                "alphabet": ALPHABET,
                "num_classes": NUM_CLASSES,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )

        mlflow.log_params(parameters)

        mlflow.log_artifact(str(args.config), artifact_path="config")

        mlflow.log_dict(config, artifact_file="config/resolved_config.json")

        best_accuracy = -1.0
        best_summary = None
        history = []

        checkpoint_path = run_dir / "best.pt"
        predictions_path = run_dir / "best_val_predictions.csv"
        history_path = run_dir / "history.json"

        for epoch in range(1, training_config["epochs"] + 1):
            started_at = perf_counter()

            if epoch == training_config["lr_switch_epoch"]:
                new_lr = training_config["new_learning_rate"]

                if training_config["reset_optimizer_on_switch"]:
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=new_lr,
                    )
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr

            learning_rate = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch}/{training_config['epochs']} | lr={learning_rate}", flush=True)

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gradient_limit=training_config["gradient_clip_norm"],
            )

            validation = validate(model=model, loader=val_loader, criterion=criterion, device=device)

            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_cer": validation["cer"],
                "learning_rate": learning_rate,
                "epoch_duration_seconds": perf_counter() - started_at,
            }

            mlflow.log_metrics(epoch_metrics, step=epoch)

            history.append({"epoch": epoch, **epoch_metrics})

            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)

            print(
                f"Train loss: {train_loss:.4f}"
                f" | Val loss: {validation['loss']:.4f}"
                f" | Accuracy: {validation['accuracy']:.2%}"
                f" | CER: {validation['cer']:.2%}",
                flush=True,
            )

            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]

                best_summary = {
                    "epoch": epoch,
                    "val_loss": validation["loss"],
                    "val_accuracy": validation["accuracy"],
                    "val_cer": validation["cer"],
                }

                checkpoint = {
                    "model_state_dict": {
                        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                    },
                    "config": config,
                    "alphabet": ALPHABET,
                    "run_id": run_id,
                    "manifest_hashes": manifest_hashes,
                    **best_summary,
                }

                torch.save(checkpoint, checkpoint_path)

                with predictions_path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["images", "target", "prediction"])
                    writer.writeheader()

                    for source, prediction in zip(val_dataset.rows, validation["predictions"], strict=True):
                        writer.writerow(
                            {
                                "images": source["images"],
                                "target": prediction["target"],
                                "prediction": prediction["prediction"],
                            }
                        )

                print("Best checkpoint saved:", checkpoint_path, flush=True)

        if best_summary is None:
            raise RuntimeError("Training did not produce a checkpoint.")

        mlflow.log_metrics(
            {
                "best_epoch": best_summary["epoch"],
                "best_val_loss": best_summary["val_loss"],
                "best_val_accuracy": best_summary["val_accuracy"],
                "best_val_cer": best_summary["val_cer"],
            }
        )

        mlflow.log_dict(best_summary, "evaluation/best_summary.json")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        mlflow.log_artifact(str(history_path), artifact_path="history")

        print(f"Best epoch: {best_summary['epoch']} | Accuracy: {best_accuracy:.2%}", flush=True)


if __name__ == "__main__":
    main()
