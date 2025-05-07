import os
from typing import List, Tuple

import numpy as np
import pertdata
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import get_git_root


def load_pseudobulk_data(
    dataset_name: str,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Load pseudobulked gene expression data.

    Supported datasets:
    - NormanWeissman2019_filtered

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Y: Data matrix with shape (n_perturbations, n_genes).
        perts: List of perturbations.
        genes: List of genes.
    """
    # Check if we support the dataset.
    if dataset_name != "NormanWeissman2019_filtered":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load the data from disk if they exist.
    artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "lgem", dataset_name)
    Y_file_path = os.path.join(artifacts_dir_path, "Y.pt")  # noqa: N806
    perts_file_path = os.path.join(artifacts_dir_path, "perts.pt")
    genes_file_path = os.path.join(artifacts_dir_path, "genes.pt")
    if os.path.exists(artifacts_dir_path):
        return (
            torch.load(Y_file_path),
            torch.load(perts_file_path),
            torch.load(genes_file_path),
        )

    # Load the dataset.
    ds = pertdata.PertDataset(
        name=dataset_name,
        cache_dir_path=os.path.join(get_git_root(), ".pertdata_cache"),
        silent=False,
    )

    # Get only single perturbations.
    adata = ds.adata[ds.adata.obs["nperts"] == 1].copy()

    # Purge perturbations from adata with missing gene expression measurements.
    genes = adata.var_names.tolist()
    perts = adata.obs["perturbation"].unique().tolist()
    adata = adata[
        adata.obs["perturbation"].isin([p for p in perts if p in genes])
    ].copy()
    perts = adata.obs["perturbation"].unique().tolist()

    # Pseudobulk the data per perturbation.
    n_perts = len(perts)
    n_genes = len(genes)
    Y = torch.zeros((n_perts, n_genes), dtype=torch.float32)  # noqa: N806
    for i, pert in tqdm(
        enumerate(perts), desc="Pseudobulking", total=n_perts, unit="perturbation"
    ):
        Y[i, :] = torch.Tensor(adata[adata.obs["perturbation"] == pert].X.mean(axis=0))

    # Save the data to disk.
    os.makedirs(artifacts_dir_path)
    torch.save(Y, Y_file_path)
    torch.save(perts, perts_file_path)
    torch.save(genes, genes_file_path)

    return Y, perts, genes


def compute_embeddings(
    Y: torch.Tensor,  # noqa: N803
    perts: List[str],
    genes: List[str],
    d_embed: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gene and perturbation embeddings.

    Args:
        Y: Data matrix with shape (n_genes, n_perturbations).
        perts: List of perturbations.
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
    P = G[np.where(np.isin(genes, perts))[0], :]  # noqa: N806

    # Compute b as the average expression of each gene across all perturbations.
    b = Y.mean(axis=1)

    return torch.from_numpy(G).float(), torch.from_numpy(P).float(), b
