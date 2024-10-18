"""Functionality for handling perturbation datasets with metabolic model integration."""

import os
import sys

import pandas as pd
import scanpy as sc

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from metabolic_utils import get_reaction_consistencies
from models import init_model_transmet
from models.gene_symbols_transmet import get_ensembl_ids_with_cache

from transmet.pert_dataset import PertDataset


class MetaPertDataset(PertDataset):
    """Class for handling a perturbation dataset with metabolic model integration.

    Inherits from PertDataset and adds functionality for integrating a metabolic model.

    Attributes:
        metabolic_model: The metabolic model associated with the dataset.
    """

    def __init__(
        self,
        name: str,
        variant: str,
        dir_path: str,
        model_name: str = None,
        metabolic_model=None,
        **kwargs,
    ) -> "MetaPertDataset":
        """Initialize the MetaPertDataset object.

        Args:
            name: The name of the dataset.
            variant: The variant of the dataset.
            dir_path: The path to the datasets directory.
            model_name: The name of the metabolic model to load
                (optional if metabolic_model is provided).
            metabolic_model: A pre-loaded metabolic model (optional).
            **kwargs: Additional arguments for initializing the metabolic model.

        Returns:
            A MetaPertDataset object.
        """
        super().__init__(name, variant, dir_path)

        if metabolic_model is not None:
            print(
                f"Using provided metabolic model for dataset '{name}' "
                f"with variant '{variant}'."
            )
            self.metabolic_model = metabolic_model
        else:
            print(
                f"Loading metabolic model '{model_name}' for dataset '{name}' "
                f"with variant '{variant}'."
            )
            self.metabolic_model = init_model_transmet(model_name, **kwargs)
            print(f"Metabolic model '{model_name}' loaded successfully.")
        # Calculate and store the perturbed genes list
        self.calculate_perturbed_genes_list()

    def calculate_perturbed_genes_list(self) -> None:
        """Calculate and store the list of perturbed genes in Ensembl IDs."""
        # Extract unique conditions
        unique_conditions = self.adata.obs["condition"].unique()

        # Initialize a set to store unique perturbed genes
        perturbed_genes = set()

        # Iterate over each unique condition
        for condition in unique_conditions:
            # Split the condition by the '+' sign
            genes = condition.split("+")
            # Discard 'ctrl' and add the remaining genes to the set
            for gene in genes:
                if gene != "ctrl":
                    perturbed_genes.add(gene)

        # Convert the set to a list
        perturbed_genes_list = list(perturbed_genes)

        # Translate gene symbols to Ensembl IDs
        ensembl_ids = get_ensembl_ids_with_cache(perturbed_genes_list)

        # Set perturbed_genes_list to the list of Ensembl IDs
        self.perturbed_genes_list = list(ensembl_ids.values())

        # Print the number of unique perturbed genes and Ensembl IDs
        print(f"Number of unique perturbed genes: {len(perturbed_genes_list)}")

    def most_variable_genes(self, n_top_genes=5000, flavor="seurat") -> None:
        """Identify the most variable genes in the dataset.

        Args:
            n_top_genes: The number of most variable genes to identify.
            flavor: The flavor of the highly variable genes calculation.
        """
        # Create a temporary log-transformed copy of the adata object
        temp_log_adata = sc.pp.log1p(self.adata, copy=True)

        # Calculate highly variable genes on the temporary log-transformed data
        sc.pp.highly_variable_genes(
            adata=temp_log_adata, n_top_genes=n_top_genes, flavor=flavor
        )

        # Transfer the highly variable genes information to the original adata object
        self.adata.var["highly_variable"] = temp_log_adata.var["highly_variable"]
        self.adata.var["means"] = temp_log_adata.var["means"]
        self.adata.var["dispersions"] = temp_log_adata.var["dispersions"]
        self.adata.var["dispersions_norm"] = temp_log_adata.var["dispersions_norm"]

        # Check if 'highly_variable_rank' exists before transferring
        if "highly_variable_rank" in temp_log_adata.var:
            self.adata.var["highly_variable_rank"] = temp_log_adata.var[
                "highly_variable_rank"
            ]

    def load_compass_results(self, file_path: str) -> None:
        """Load COMPASS metabolic analysis results from a file into the dataset.

        Args:
            file_path: The path to the file containing COMPASS results (reactions.tsv).
        """
        # Read the COMPASS results from the file
        compass_results = pd.read_csv(file_path, sep="\t", index_col=0)

        # Convert the DataFrame to a dictionary
        self.compass_results = get_reaction_consistencies(
            compass_results, min_range=0.0
        )

        print("COMPASS metabolic analysis results loaded successfully.")

    def calculate_reaction_stats(self) -> None:
        """Calculate the mean and variance of reaction penalty scores."""
        # Calculate mean and variance for each reaction across rows
        mean_scores = self.compass_results.mean(axis=1)
        variance_scores = self.compass_results.var(axis=1)

        # Convert keys to upper case
        mean_scores = mean_scores.rename(index=str.upper)
        variance_scores = variance_scores.rename(index=str.upper)

        # Store the results in the reaction_stats attribute
        self.reaction_stats = {
            "mean": mean_scores.to_dict(),
            "variance": variance_scores.to_dict(),
        }
        print("Reaction penalty scores mean and variance calculated successfully.")
