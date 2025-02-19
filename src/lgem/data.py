import os
from typing import List, Tuple

import numpy as np
import pertdata
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from .utils import get_git_root


def load_data(
    dataset_name: str,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Load gene expression data.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Y: Data matrix with shape (n_perturbations, n_genes).
        perturbations: List of perturbations.
        genes: List of genes.
    """
    # Check if we support the dataset.
    if dataset_name != "NormanWeissman2019_filtered":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load the data from disk if they exist.
    artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "lgem", dataset_name)
    if os.path.exists(artifacts_dir_path):
        Y = torch.load(os.path.join(artifacts_dir_path, "Y.pt"))  # noqa: N806
        perturbations = torch.load(os.path.join(artifacts_dir_path, "perturbations.pt"))
        genes = torch.load(os.path.join(artifacts_dir_path, "genes.pt"))
        return Y, perturbations, genes

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
    Y = np.zeros((n_perturbations, n_genes))  # noqa: N806
    for i, pert in tqdm(
        enumerate(perturbations),
        desc="Pseudobulking",
        total=n_perturbations,
        unit="perturbation",
    ):
        Y[i, :] = adata[adata.obs["perturbation"] == pert].X.mean(axis=0)
    Y = torch.from_numpy(Y).float()  # noqa: N806

    # Save the data to disk.
    os.makedirs(artifacts_dir_path)
    torch.save(Y, os.path.join(artifacts_dir_path, "Y.pt"))
    torch.save(perturbations, os.path.join(artifacts_dir_path, "perturbations.pt"))
    torch.save(genes, os.path.join(artifacts_dir_path, "genes.pt"))

    return Y, perturbations, genes


def compute_embeddings(
    Y: torch.Tensor,  # noqa: N803
    perturbations: List[str],
    genes: List[str],
    d_embed: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gene and perturbation embeddings.

    Args:
        Y: Data matrix with shape (n_genes, n_perturbations).
        perturbations: List of perturbations.
        genes: List of genes.
        d_embed: Embedding dimension.

    Returns:
        G: Gene embedding matrix with shape (n_genes, d_embed).
        P: Perturbation embedding matrix with shape (n_perturbations, d_embed).
        b: Bias vector with shape (n_genes).
    """
    # Perform a PCA on Y to obtain the top d_embed principal components, which will
    # serve as the gene embeddings G.
    pca = PCA(n_components=d_embed)
    G = pca.fit_transform(Y)  # noqa: N806

    # Extract perturbation embeddings P from G by subsetting G to only those rows
    # corresponding to genes that have been perturbed in the data.
    P = G[np.where(np.isin(genes, perturbations))[0], :]  # noqa: N806

    # Compute b as the average expression of each gene across all perturbations.
    b = Y.mean(axis=1)

    return torch.from_numpy(G).float(), torch.from_numpy(P).float(), b
