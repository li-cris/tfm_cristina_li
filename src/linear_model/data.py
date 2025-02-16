import os  # noqa: D100
from typing import List, Tuple

import numpy as np
import pertdata
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import get_git_root


def load_data(  # noqa: D103
    dataset_name: str,
) -> Tuple[
    torch.Tensor,  # Y_train with shape (n_perturbations, n_genes)
    List[str],  # perturbations
    List[str],  # genes
]:
    # Check if we support the dataset.
    if dataset_name != "NormanWeissman2019_filtered":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load the data from disk if it exists.
    artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "linear_model", dataset_name)
    if os.path.exists(artifacts_dir_path):
        Y_train = torch.load(os.path.join(artifacts_dir_path, "Y_train.pt"))  # noqa: N806
        perturbations = torch.load(os.path.join(artifacts_dir_path, "perturbations.pt"))
        genes = torch.load(os.path.join(artifacts_dir_path, "genes.pt"))
        return Y_train, perturbations, genes
    os.makedirs(artifacts_dir_path)

    # Load the dataset.
    ds = pertdata.PertDataset(
        name=dataset_name, cache_dir_path=os.path.join(get_git_root(), ".pertdata_cache"), silent=False
    )
    adata = ds.adata

    # Demultiplex adata for single perturbations.
    adata = adata[adata.obs["nperts"] == 1]

    # Remove perturbations with missing gene expression values.
    genes = adata.var_names.tolist()
    perturbations = adata.obs["perturbation"].unique()
    perturbations = [p for p in perturbations if p in adata.var_names]
    adata = adata[adata.obs["perturbation"].isin(perturbations)]

    # Pseudobulk the data per perturbation.
    n_perturbations = len(perturbations)
    n_genes = len(genes)
    Y_train = np.zeros((n_perturbations, n_genes))  # noqa: N806
    for i, pert in tqdm(enumerate(perturbations), desc="Pseudobulking", total=n_perturbations, unit="perturbation"):
        Y_train[i, :] = adata[adata.obs["perturbation"] == pert].X.mean(axis=0)
    Y_train = torch.from_numpy(Y_train).float()  # noqa: N806

    # Save the data to disk.
    torch.save(Y_train, os.path.join(artifacts_dir_path, "Y_train.pt"))
    torch.save(perturbations, os.path.join(artifacts_dir_path, "perturbations.pt"))
    torch.save(genes, os.path.join(artifacts_dir_path, "genes.pt"))

    return Y_train, perturbations, genes


def compute_embeddings(  # noqa: D103
    Y_train: torch.Tensor,  # (n_perturbations, n_genes)  # noqa: N803
    perturbations: List[str],
    genes: List[str],
    d_embed: int = 10,
) -> Tuple[
    torch.Tensor,  # G with shape (n_genes, d_embed)
    torch.Tensor,  # P with shape (n_perturbations, d_embed)
    torch.Tensor,  # b with shape (n_perturbations)
]:
    # Perform a PCA on Y_train.T to obtain the top d_embed principal components, which
    # will serve as the gene embeddings G.
    pca = PCA(n_components=d_embed)
    G = pca.fit_transform(Y_train.T)  # noqa: N806

    # Extract perturbation embeddings P from G by subsetting G to only those rows
    # corresponding to genes that have been perturbed in the training data.
    P = G[np.where(np.isin(genes, perturbations))[0], :]  # noqa: N806

    # Compute b as the average expression of each gene across all perturbations.
    b = Y_train.mean(axis=0)

    # Convert the NumPy arrays to PyTorch tensors.
    G = torch.from_numpy(G).float()  # noqa: N806
    P = torch.from_numpy(P).float()  # noqa: N806

    return G, P, b


def create_dataloaders(  # noqa: D103
    Y_train: torch.Tensor,  # noqa: N803
    P: torch.Tensor,  # noqa: N803
    batch_size: int,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = TensorDataset(P, Y_train)

    # Calculate the sizes of each split.
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    # Split the dataset.
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create the dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
