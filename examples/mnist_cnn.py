"""
CNN on MNIST with data augmentation, weight decay, early stopping, and inference.

Run from the examples/ directory: python mnist_cnn.py
MNIST is downloaded to ./data on the first run. Logs and checkpoints go to ./outputs.
"""
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kito import Engine, KitoModule
from kito.callbacks import EarlyStoppingCallback
from kito.config.moduleconfig import KitoModuleConfig, ModelConfig, TrainingConfig, WorkDirConfig


class CNNMNISTModel(KitoModule):
    def build_inner_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
        self.model_input_size = (1, 28, 28)

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)


def main():
    config = KitoModuleConfig(
        training=TrainingConfig(learning_rate=1e-3, n_train_epochs=20, batch_size=128, train_mode=True),
        model=ModelConfig(
            module_name='mnist_cnn',
            input_data_size=(1, 28, 28),
            loss='cross_entropy_loss',
            train_codename='mnist_cnn',
        ),
        workdir=WorkDirConfig(work_directory='./outputs'),
    )

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=config.training.batch_size)

    # Training: stops early when val_loss has not improved for 3 epochs
    engine = Engine(CNNMNISTModel(config), config)
    engine.fit(train_loader, val_loader, callbacks=[EarlyStoppingCallback(monitor='val_loss', patience=3)])

    # Inference with the most recent checkpoint
    checkpoints = glob.glob('./outputs/weights/mnist_cnn/*.pt')
    config.training.train_mode = False
    config.model.weight_load_path = max(checkpoints, key=os.path.getmtime)
    engine = Engine(CNNMNISTModel(config), config)

    images = torch.stack([test_set[i] for i in range(1000)])
    labels = torch.tensor([test_set[i] for i in range(1000)])[4]
    logits = engine.predict(images)  # NumPy array, shape (1000, 10)
    accuracy = (logits.argmax(axis=1) == labels.numpy()).mean()
    print(f'Accuracy on the first 1000 test images: {accuracy:.3f}')


if __name__ == '__main__':
    main()
