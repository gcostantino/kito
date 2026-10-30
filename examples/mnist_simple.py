"""
Minimal MNIST example: a feed-forward classifier trained with Kito, then used for inference.

Run from the examples/ directory: python mnist_simple.py
MNIST is downloaded to ./data on the first run. Logs and checkpoints go to ./outputs.
"""
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from kito import Engine, KitoModule
from kito.config.moduleconfig import KitoModuleConfig, ModelConfig, TrainingConfig, WorkDirConfig


class SimpleMNISTModel(KitoModule):
    def build_inner_model(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        self.model_input_size = (1, 28, 28)

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


def main():
    config = KitoModuleConfig(
        training=TrainingConfig(learning_rate=1e-3, n_train_epochs=10, batch_size=64, train_mode=True),
        model=ModelConfig(
            module_name='mnist_simple',
            input_data_size=(1, 28, 28),
            loss='cross_entropy_loss',
            train_codename='mnist_simple',
        ),
        workdir=WorkDirConfig(work_directory='./outputs'),
    )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=config.training.batch_size)

    # Training: the test set serves as validation data here
    engine = Engine(SimpleMNISTModel(config), config)
    engine.fit(train_loader, val_loader)

    # Inference: same config with train_mode=False and the most recent checkpoint
    checkpoints = glob.glob('./outputs/weights/mnist_simple/*.pt')
    config.training.train_mode = False
    config.model.weight_load_path = max(checkpoints, key=os.path.getmtime)
    engine = Engine(SimpleMNISTModel(config), config)

    images = torch.stack([test_set[i] for i in range(1000)])
    labels = torch.tensor([test_set[i] for i in range(1000)])[4]
    logits = engine.predict(images)  # NumPy array, shape (1000, 10)
    accuracy = (logits.argmax(axis=1) == labels.numpy()).mean()
    print(f'Accuracy on the first 1000 test images: {accuracy:.3f}')


if __name__ == '__main__':
    main()
