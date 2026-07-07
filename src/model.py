import torch
import torch.nn as nn

class SeizureCNN(nn.Module):
    def __init__(self, n_channels=23, n_timepoints=1280):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.relu = nn.ReLU()

        # Figure out flattened size after convolutions+pooling
        reduced_timepoints = n_timepoints // 4 // 4  # two pool layers, each /4
        self.flatten_size = 32 * reduced_timepoints

        # Classifier head
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)  # single output: seizure probability (logit)

    def forward(self, x):
        # x shape: (batch_size, n_channels, n_timepoints)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.flatten(start_dim=1)  # flatten everything except batch dimension

        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # raw logit, no sigmoid here (we'll apply it in the loss function)

        return x


class SeizureCNN2(nn.Module):
    """Regularized CNN aimed at cross-subject generalization.

    Differences from SeizureCNN that matter for generalizing to unseen patients:
      * BatchNorm after each conv (stabilizes across patients with different scales)
      * Global average pooling instead of flatten -> large FC (far fewer params,
        so it memorizes the training patients less)
      * Dropout before the classifier
    """

    def __init__(self, n_channels=23, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 64, 1), length-agnostic
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = SeizureCNN()
    print(model)

    # Sanity check: pass a fake batch through it
    fake_batch = torch.randn(4, 23, 1280)  # batch of 4 windows
    output = model(fake_batch)
    print("Output shape:", output.shape)  # should be (4, 1)