"""
CNN MNIST example with Kito.

Demonstrates a convolutional neural network with callbacks.
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
from kito.callbacks import ModelCheckpoint, CSVLogger, TextLogger


class CNNMNISTModel(KitoModule):
    """Convolutional neural network for MNIST."""

    def build_inner_model(self):
        """Build CNN architecture."""
        self.model = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Classifier
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self.model_input_size = (1, 28, 28)

    def bind_optimizer(self):
        """Use Adam with weight decay."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )


def get_mnist_loaders(batch_size=64):
    """Load MNIST with data augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=2
    )

    return train_loader, test_loader


def main():
    """Train CNN MNIST model with callbacks."""

    # Configuration
    config = KitoModuleConfig(
        training=TrainingConfig(
            learning_rate=1e-3,
            n_train_epochs=20,
            batch_size=128,
            distributed_training=False,
        ),
        model=ModelConfig(
            loss='cross_entropy_loss',
            save_model_weights=True,
            train_codename='mnist_cnn',
            log_to_tensorboard=False,  # Set True if you have tensorboard installed
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
    model = CNNMNISTModel('MNIST-CNN', device, config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath='./outputs/weights/best_mnist_cnn.pt',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=True
        ),
        CSVLogger('./outputs/logs/training.csv'),
        TextLogger('./outputs/logs/training.log')
    ]

    # Engine
    engine = Engine(model, config)

    # Train with callbacks
    print("\nStarting training...")
    engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.training.n_train_epochs,
        callbacks=callbacks
    )

    print("\n✅ Training complete!")
    print(f"Best model saved to: ./outputs/weights/best_mnist_cnn.pt")
    print(f"Training logs saved to: ./outputs/logs/")


if __name__ == '__main__':
    main()
