"""This module defines a Multi-Layer Perceptron (MLP) model."""

import torch
from torch import nn

import pytorch_lightning as pl


class MLP(pl.LightningModule):
    """Simple MLP model."""

    def __init__(
        self, in_features: int, out_features: int, learning_rate: float = 1e-3
    ) -> None:
        """
        Initialize the MLP model.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            learning_rate: The learning rate for the optimizer. Default is 1e-3.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_features),
        )

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):  # noqa: D102
        return self.layers(x)

    def training_step(self, batch, batch_idx):  # noqa: D102
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log(name="batch_idx", value=int(batch_idx), prog_bar=True)
        self.log(name="train_loss", value=loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
