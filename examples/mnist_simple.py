"""
Simple MNIST example with Kito.

Demonstrates the minimal code needed to train a model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kito import Engine, KitoModule
from kito.config.moduleconfig import (
    KitoModuleConfig,
    TrainingConfig,
    ModelConfig,
    DataConfig,
    WorkDirConfig
)


class SimpleMNISTModel(KitoModule):
    """Simple feedforward network for MNIST."""

    def build_inner_model(self):
        """Build a simple 2-layer network."""
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        self.model_input_size = (1, 28, 28)  # MNIST image size

    def bind_optimizer(self):
        """Use Adam optimizer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )


def get_mnist_loaders(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size
    )

    return train_loader, test_loader


def main():
    """Train MNIST model with Kito."""

    # Configuration
    config = KitoModuleConfig(
        training=TrainingConfig(
            learning_rate=1e-3,
            n_train_epochs=10,
            batch_size=64,
            distributed_training=False,
        ),
        model=ModelConfig(
            loss='cross_entropy_loss',
            save_model_weights=True,
            train_codename='mnist_simple',
        ),
        data=DataConfig(),
        workdir=WorkDirConfig(
            work_directory='./outputs'
        )
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_mnist_loaders(batch_size=config.training.batch_size)

    # Model
    model = SimpleMNISTModel('MNIST-Simple', device, config)

    # Engine
    engine = Engine(model, config)

    # Train!
    print("\nStarting training...")
    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.training.n_train_epochs
    )

    print("\n✅ Training complete!")
    print(f"Model weights saved to: ./outputs/weights/")


if __name__ == '__main__':
    main()
