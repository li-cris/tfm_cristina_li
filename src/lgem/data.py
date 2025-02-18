import os
from typing import List, Tuple

import numpy as np
import pertdata
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import get_git_root


def load_data(
    dataset_name: str,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Load gene expression data.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Y_train: Data matrix with shape (n_genes, n_perturbations).
        perturbations: List of perturbations.
        genes: List of genes.
    """
    # Check if we support the dataset.
    if dataset_name != "NormanWeissman2019_filtered":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load the data from disk if they exist.
    artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "lgem", dataset_name)
    if os.path.exists(artifacts_dir_path):
        Y_train = torch.load(os.path.join(artifacts_dir_path, "Y_train.pt"))  # noqa: N806
        perturbations = torch.load(os.path.join(artifacts_dir_path, "perturbations.pt"))
        genes = torch.load(os.path.join(artifacts_dir_path, "genes.pt"))
        return Y_train, perturbations, genes

    # Load the dataset.
    ds = pertdata.PertDataset(
        name=dataset_name,
        cache_dir_path=os.path.join(get_git_root(), ".pertdata_cache"),
        silent=False,
    )

    # Get only single perturbations from adata.
    adata = ds.adata
    adata = adata[adata.obs["nperts"] == 1]

    # Remove perturbations with missing gene expression values.
    genes = adata.var_names.tolist()
    perturbations = adata.obs["perturbation"].unique().tolist()
    perturbations = [p for p in perturbations if p in adata.var_names]
    adata = adata[adata.obs["perturbation"].isin(perturbations)]

    # Pseudobulk the data per perturbation.
    n_perturbations = len(perturbations)
    n_genes = len(genes)
    Y_train = np.zeros((n_genes, n_perturbations))  # noqa: N806
    for i, pert in tqdm(
        enumerate(perturbations),
        desc="Pseudobulking",
        total=n_perturbations,
        unit="perturbation",
    ):
        Y_train[:, i] = adata[adata.obs["perturbation"] == pert].X.mean(axis=0)
    Y_train = torch.from_numpy(Y_train).float()  # noqa: N806

    # Save the data to disk.
    os.makedirs(artifacts_dir_path)
    torch.save(Y_train, os.path.join(artifacts_dir_path, "Y_train.pt"))
    torch.save(perturbations, os.path.join(artifacts_dir_path, "perturbations.pt"))
    torch.save(genes, os.path.join(artifacts_dir_path, "genes.pt"))

    return Y_train, perturbations, genes


def compute_embeddings(
    Y_train: torch.Tensor,  # noqa: N803
    perturbations: List[str],
    genes: List[str],
    d_embed: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gene and perturbation embeddings.

    Args:
        Y_train: Data matrix with shape (n_genes, n_perturbations).
        perturbations: List of perturbations.
        genes: List of genes.
        d_embed: Embedding dimension.

    Returns:
        G: Gene embedding matrix with shape (n_genes, d_embed).
        P: Perturbation embedding matrix with shape (n_perturbations, d_embed).
        b: Bias vector with shape (n_genes).
    """
    # Perform a PCA on Y_train to obtain the top d_embed principal components, which
    # will serve as the gene embeddings G.
    pca = PCA(n_components=d_embed)
    G = pca.fit_transform(Y_train)  # noqa: N806

    # Extract perturbation embeddings P from G by subsetting G to only those rows
    # corresponding to genes that have been perturbed in the training data.
    P = G[np.where(np.isin(genes, perturbations))[0], :]  # noqa: N806

    # Compute b as the average expression of each gene across all perturbations.
    b = Y_train.mean(axis=1)

    return torch.from_numpy(G).float(), torch.from_numpy(P).float(), b


def create_dataloaders(
    Y_train: torch.Tensor,  # noqa: N803
    P: torch.Tensor,  # noqa: N803
    batch_size: int,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing.

    Args:
        Y_train: Data matrix with shape (n_genes, n_perturbations).
        P: Perturbation embedding matrix with shape (n_perturbations, d_embed).
        batch_size: Batch size.
        val_split: Fraction of the dataset to include in the validation set.
        test_split: Fraction of the dataset to include in the test set.

    Returns:
        train_dataloader: Dataloader for the training set.
        val_dataloader: Dataloader for the validation set.
        test_dataloader: Dataloader for the test set.
    """
    dataset = TensorDataset(P, Y_train.T)

    # Calculate the sizes of each split.
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    # Split the dataset.
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create the dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
