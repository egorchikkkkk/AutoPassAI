from torch import nn

from autopassai.ml.charset import NUM_CLASSES


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=128):
        super().__init__()

        self.cnn = nn.Sequential(
            # [B, 1, 32, 160] → [B, 32, 16, 80]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # → [B, 64, 8, 40]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # → [B, 128, 4, 40]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # → [B, 256, 2, 40]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # → [B, 256, 1, 40]
            nn.Conv2d(256, 256, kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, images):
        features = self.cnn(images)
        features = features.squeeze(2)
        sequence = features.permute(0, 2, 1)

        sequence, _ = self.rnn(sequence)
        logits = self.classifier(sequence)

        # CTC waits: steps, batch, classes
        return logits.permute(1, 0, 2)  # [40, B, 35]
