from PIL import Image
from torchvision import transforms


class ResizeAndPad:
    def __init__(self, height=32, width=160):
        self.height = height
        self.width = width

    def __call__(self, image):
        original_width, original_height = image.size

        # Scale with proportion saving
        scale = min(
            self.width / original_width,
            self.height / original_height,
        )

        new_width = max(1, round(original_width * scale))
        new_height = max(1, round(original_height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        # White background, image in left (horizontal) center (vertical)
        canvas = Image.new("L", (self.width, self.height), color=255)
        top = (self.height - new_height) // 2
        canvas.paste(image, (0, top))

        return canvas


def build_transform(mode: str, height=32, width=160):
    if height != 32:
        raise ValueError("Для текущей CRNN поддерживается только height=32.")

    if width <= 0:
        raise ValueError("Ширина должна быть положительной.")

    if mode == "pad":
        resize = ResizeAndPad(height, width)

    elif mode == "resize":
        resize = transforms.Resize(
            (height, width),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )

    else:
        raise ValueError(f"Неизвестный режим {mode!r}. Ожидается 'pad' или 'resize'.")

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
