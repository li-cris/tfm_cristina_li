from typing import Dict, Tuple

import git
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_git_root() -> str:
    """Get the root directory of the Git repository."""
    return git.Repo(search_parent_directories=True).working_tree_dir


def load_gene_pathway_mask(
    map_file_path: str,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Load gene-pathway mask from file.

    The pathway-gene map file must be a tab-separated file with two columns:
    pathway_id and ensembl_gene_id.

    Args:
        map_file_path: Path to pathway-gene map file.

    Returns:
        mask: Gene-pathway mask with shape (n_genes, n_pathways).
        gene_to_index: Mapping from gene to index.
        pathway_to_index: Mapping from pathway to index.
    """
    # Load the pathway-gene map file.
    df = pd.read_csv(map_file_path, sep="\t")

    # Strip any leading or trailing whitespace from the columns.
    df["pathway_id"] = df["pathway_id"].str.strip()
    df["ensembl_gene_id"] = df["ensembl_gene_id"].str.strip()

    # Get unique genes and pathways.
    genes = df["ensembl_gene_id"].unique()
    pathways = df["pathway_id"].unique()

    # Create a mapping from gene to index and pathway to index.
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    pathway_to_index = {pathway: idx for idx, pathway in enumerate(pathways)}

    # Initialize the mask matrix.
    mask = np.zeros((len(genes), len(pathways)), dtype=int)

    # Fill the mask matrix.
    for _, row in tqdm(
        df.iterrows(), desc="Loading gene-pathway mask", total=len(df), unit="pathway"
    ):
        gene_idx = gene_to_index[row["ensembl_gene_id"]]
        pathway_idx = pathway_to_index[row["pathway_id"]]
        mask[gene_idx, pathway_idx] = 1

    return mask, gene_to_index, pathway_to_index
