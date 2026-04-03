"""
Neural network architectures for the ML Training Optimizer environment.

Provides three model architectures of increasing complexity:
- SimpleMLP: For MNIST (Easy task)
- SmallCNN: For FashionMNIST (Medium task)
- DeeperCNN: For CIFAR-10 (Hard task)

All models accept a configurable dropout rate set by the agent.
"""

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    2-layer MLP for MNIST digit classification.
    ~100k parameters. Trains in ~0.5-1s per epoch on CPU with 4k samples.

    Architecture: 784 → 128 → 64 → 10
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class SmallCNN(nn.Module):
    """
    Small CNN for FashionMNIST classification.
    ~200k parameters. Trains in ~1-2s per epoch on CPU with 6.5k samples.

    Architecture: 2×(Conv→ReLU→MaxPool) → FC(128) → FC(10)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DeeperCNN(nn.Module):
    """
    Deeper CNN for CIFAR-10 classification.
    ~500k parameters. Trains in ~2-3s per epoch on CPU with 8k samples.

    Architecture: 3×(Conv→BN→ReLU→MaxPool) → FC(256) → FC(10)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→32, 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32→64, 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64→128, 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(model_type: str, dropout: float = 0.0) -> nn.Module:
    """Factory function to create a model by name."""
    models = {
        "simple_mlp": SimpleMLP,
        "small_cnn": SmallCNN,
        "deeper_cnn": DeeperCNN,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type](dropout=dropout)
