"""This module provides functionality to train and test an MLP on perturbation data."""

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd

import torch

import pytorch_lightning as pl

from causal_hts_modeling.models import MLP
from causal_hts_modeling.pertdata import PertData
from causal_hts_modeling.utils import log_config, get_git_root

log = logging.getLogger(__name__)


def train_and_test_mlp(cfg: DictConfig) -> None:
    """
    Train and test an MLP.

    Args:
        cfg: Hydra configuration.
    """
    # Create a PertData object for loading and processing perturbation data
    pert_data = PertData(
        data_dir=f"{get_git_root()}/{cfg.data.data_dir}",
        dataset_name=cfg.data.dataset_name,
        fix_labels=True,
    )
    pert_data.log_info()

    # Filter out everything except for single-gene perturbations
    pert_data.filter_only_single_gene_perturbations_()
    pert_data.log_info()

    # Convert the perturbations vector first to a numerical categorical representation
    # and then to a PyTorch tensor
    y = pd.factorize(values=pert_data.y)[0]
    y = torch.tensor(data=y, dtype=torch.long)

    # Convert the gene expression matrix to a PyTorch tensor
    X = torch.tensor(data=pert_data.X.toarray(), dtype=torch.float32)

    # Create a PyTorch dataset and a dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32, shuffle=True
    )

    # Create an MLP
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    log.info(f"Number of features: {num_features}")
    log.info(f"Number of classes: {num_classes}")
    mlp = MLP(in_features=num_features, out_features=num_classes)

    # Train the MLP
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=mlp, train_dataloaders=train_loader)


@hydra.main(version_base=None, config_path="configs", config_name="mlp_playground")
def main(cfg: DictConfig) -> None:
    """
    Run the MLP playground.

    Args:
        cfg: Hydra configuration.
    """
    try:
        log_config(cfg=cfg)
        log.info(f"User: {cfg.info.user}")
        log.info(f"Git root: {get_git_root()}")
        device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {device}")

        train_and_test_mlp(cfg=cfg)
    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
