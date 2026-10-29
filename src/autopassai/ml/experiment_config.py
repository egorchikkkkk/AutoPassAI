import argparse
import json
import math
import tomllib
from pathlib import Path

REQUIRED_FIELDS = {
    "experiment": {"name", "run_name"},
    "preprocessing": {"mode", "height", "width"},
    "model": {"hidden_size"},
    "training": {
        "seed",
        "batch_size",
        "epochs",
        "num_workers",
        "learning_rate",
        "lr_switch_epoch",
        "new_learning_rate",
        "reset_optimizer_on_switch",
        "gradient_clip_norm",
        "checkpoint_metric",
        "checkpoint_mode",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: Path) -> dict:
    with Path(path).open("rb") as file:
        config = tomllib.load(file)

    # Check section appearance
    expected_sections = set(REQUIRED_FIELDS)
    actual_sections = set(config)

    require(
        actual_sections == expected_sections,
        f"Неверные секции. "
        f"Отсутствуют: {sorted(expected_sections - actual_sections)}. "
        f"Неизвестные: {sorted(actual_sections - expected_sections)}.",
    )

    # Check section fields
    for section, expected_fields in REQUIRED_FIELDS.items():
        values = config[section]

        require(
            isinstance(values, dict),
            f"{section}: ожидалась TOML-секция.",
        )

        actual_fields = set(values)

        require(
            actual_fields == expected_fields,
            f"{section}: "
            f"отсутствуют поля {sorted(expected_fields - actual_fields)}; "
            f"неизвестные поля {sorted(actual_fields - expected_fields)}.",
        )

    experiment = config["experiment"]
    preprocessing = config["preprocessing"]
    model = config["model"]
    training = config["training"]

    for field in ("name", "run_name"):
        value = experiment[field]
        require(
            isinstance(value, str) and bool(value.strip()),
            f"experiment.{field}: ожидалась непустая строка.",
        )

    require(
        preprocessing["mode"] in ("pad", "resize"),
        "preprocessing.mode: допустимы только 'pad' и 'resize'.",
    )

    for field, expected in (("height", 32), ("width", 160)):
        value = preprocessing[field]
        require(
            type(value) is int and value == expected,
            f"preprocessing.{field}: ожидается целое число {expected}.",
        )

    # Set minimal values for integers
    integer_fields = [
        ("model.hidden_size", model["hidden_size"], 1),
        ("training.seed", training["seed"], 0),
        ("training.batch_size", training["batch_size"], 1),
        ("training.epochs", training["epochs"], 1),
        ("training.num_workers", training["num_workers"], 0),
        ("training.lr_switch_epoch", training["lr_switch_epoch"], 1),
    ]

    for name, value, minimum in integer_fields:
        require(
            type(value) is int and value >= minimum,
            f"{name}: ожидается целое число не меньше {minimum}.",
        )

    for field in (
        "learning_rate",
        "new_learning_rate",
        "gradient_clip_norm",
    ):
        value = training[field]
        require(
            type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"training.{field}: ожидается положительное конечное число.",
        )

    require(
        training["lr_switch_epoch"] <= training["epochs"],
        "training.lr_switch_epoch не может превышать training.epochs.",
    )

    require(
        type(training["reset_optimizer_on_switch"]) is bool,
        "training.reset_optimizer_on_switch: ожидается true или false.",
    )

    require(
        training["checkpoint_metric"] == "val_accuracy",
        "training.checkpoint_metric: поддерживается только 'val_accuracy'.",
    )

    require(
        training["checkpoint_mode"] == "max",
        "training.checkpoint_mode: поддерживается только 'max'.",
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="Чтение и проверка конфигурации эксперимента.")
    parser.add_argument("config", type=Path, help="Путь к TOML-файлу.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("Конфигурация корректна:")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
