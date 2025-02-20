import os
from typing import Tuple

import numpy as np
import pandas as pd
import pertdata
import scanpy as sc
from anndata import AnnData

from .utils import get_git_root, load_gene_pathway_mask


def load_data(
    dataset_name: str,
    gene_info_file_path: str = os.path.join(
        get_git_root(), "resources", "ensembl_gene_info.tsv"
    ),
    go_gene_map_file_path: str = os.path.join(
        get_git_root(), "resources", "sena_go_gene_map.tsv"
    ),
    n_top_genes: int = 200,
    min_genes_per_pathway: int = 5,
) -> Tuple[AnnData, np.ndarray]:
    if dataset_name != "norman":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create a dictionary to map gene names to Ensembl IDs.
    gene_info = pd.read_csv(gene_info_file_path, sep="\t")
    gene_name_to_ensembl_id = dict(
        zip(gene_info["external_gene_name"], gene_info["ensembl_gene_id"])
    )

    # Load the dataset.
    ds = pertdata.PertDataset(
        name=dataset_name,
        cache_dir_path=os.path.join(get_git_root(), ".pertdata_cache"),
        silent=False,
    )
    adata = ds.adata
    print(f"Loaded dataset: {dataset_name}")
    print(f"  Shape (n_obs, n_vars): {adata.shape}")

    # Filter the dataset for single perturbations only.
    adata = adata[adata.obs["condition"].str.contains("ctrl")].copy()
    print("Filtered for single perturbations.")
    print(f"  New shape (n_obs, n_vars): {adata.shape}")

    # Fix the perturbation names.
    adata.obs["perturbation"] = adata.obs["condition"].str.replace(
        r"\+ctrl$", "", regex=True
    )
    adata.obs["perturbation"] = adata.obs["perturbation"].str.replace(
        r"^ctrl\+", "", regex=True
    )

    # Remove ctrl.
    adata = adata[~adata.obs["perturbation"].str.contains("ctrl")].copy()

    ####################################################################################
    adata = adata[: int(0.01 * adata.shape[0])].copy()  # TODO: Remove this.
    ####################################################################################

    # Add new adata.obs["perturbation_ensembl_id"] with Ensembl IDs. Remove
    # perturbations with missing Ensembl IDs.
    adata.obs["perturbation_ensembl_id"] = adata.obs["perturbation"].map(
        gene_name_to_ensembl_id
    )
    n_unique_perturbations_before = adata.obs["perturbation"].nunique()
    adata = adata[adata.obs["perturbation_ensembl_id"].notna()].copy()
    n_unique_perturbations_after = adata.obs["perturbation_ensembl_id"].nunique()
    print("Removed perturbations with missing Ensembl IDs.")
    print(f"  New shape (n_obs, n_vars): {adata.shape}")
    print(f"  Before: {n_unique_perturbations_before} unique perturbations")
    print(f"  After: {n_unique_perturbations_after} unique perturbations")

    # Filter adata.var_names to keep only those that are in the values of
    # gene_name_to_ensembl_id.
    valid_ensembl_ids = set(gene_name_to_ensembl_id.values())
    adata = adata[:, adata.var_names.isin(valid_ensembl_ids)].copy()
    print("Removed genes with missing Ensembl IDs.")
    print(f"  New shape (n_obs, n_vars): {adata.shape}")

    # Load gene-pathway mask.
    gene_pathway_mask, gene_to_index, pathway_to_index = load_gene_pathway_mask(
        go_gene_map_file_path
    )
    print("Loaded gene-pathway mask.")
    print(f"  Shape (n_genes, n_pathways): {gene_pathway_mask.shape}")

    # Remove genes from adata.var_names that are not in the gene-pathway mask.
    adata = adata[:, adata.var_names.isin(gene_to_index.keys())].copy()
    print("Removed genes not in the gene-pathway mask.")
    print(f"  New shape (n_obs, n_vars): {adata.shape}")

    # Remove perturbations with missing gene expression measurements.
    unique_perturbations = adata.obs["perturbation_ensembl_id"].unique().tolist()
    keep_perturbations = [p for p in unique_perturbations if p in adata.var_names]
    adata = adata[adata.obs["perturbation_ensembl_id"].isin(keep_perturbations)].copy()
    print("Removed perturbations with missing gene expression measurements.")
    print(f"  New shape (n_obs, n_vars): {adata.shape}")
    print(f"  Before: {len(unique_perturbations)} unique perturbations")
    print(f"  After: {len(keep_perturbations)} unique perturbations")

    # Filter genes with low variance.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    # Reduce gene-pathway mask to only include genes in adata.var_names.
    gene_indexes_to_keep = [gene_to_index[gene] for gene in adata.var_names]
    gene_pathway_mask = gene_pathway_mask[gene_indexes_to_keep, :]
    print("Reduced gene-pathway mask to only include measured genes.")
    print(f"  New shape (n_genes, n_pathways): {gene_pathway_mask.shape}")

    # Reduce gene-pathway mask to only include pathways with at least
    # min_genes_per_pathway associated genes.
    non_zero_counts = np.count_nonzero(gene_pathway_mask, axis=0)
    pathway_indexes_to_keep = np.where(non_zero_counts >= min_genes_per_pathway)[0]
    gene_pathway_mask = gene_pathway_mask[:, pathway_indexes_to_keep]
    print(
        f"Reduced gene-pathway mask to only include pathways with at least "
        f"{min_genes_per_pathway} associated genes."
    )
    print(f"  New shape (n_genes, n_pathways): {gene_pathway_mask.shape}")

    return adata, gene_pathway_mask
