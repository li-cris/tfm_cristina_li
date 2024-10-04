"""Functionality for handling perturbation data."""

import os
import sys
from typing import Optional, List

from anndata import AnnData
import gdown
import pandas as pd
import scanpy as sc

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transmet.utils import download_file, extract_zip, get_git_root


class PertData:
    """
    Class for perturbation data.

    The perturbation data is stored in an `AnnData` object. Refer to
    https://anndata.readthedocs.io/en/latest/ for more information on `AnnData`.

    `AnnData` is specifically designed for matrix-like data. By this we mean that we
    have n observations, each of which can be represented as d-dimensional vectors,
    where each dimension corresponds to a variable or feature. Both the rows and columns
    of this matrix are special in the sense that they are indexed.

    For instance, in scRNA-seq data:
    - Each row corresponds to a cell with a cell identifier (i.e., barcode).
    - Each column corresponds to a gene with a gene identifier.

    Attributes:
        name: The name of the dataset.
        variant: The variant of the dataset.
        path: The path where the dataset is stored.
        adata: The actual perturbation data.
    """

    def __init__(self) -> None:
        """Initialize the PertData object."""
        self.name: Optional[str] = None
        self.variant: Optional[str] = None
        self.path: Optional[str] = None
        self.adata: Optional[AnnData] = None

    def __str__(self) -> str:
        """Return a string representation of the PertData object."""
        return (
            f"PertData object\n"
            f"    name: {self.name}\n"
            f"    variant: {self.variant}\n"
            f"    path: {self.path}\n"
            f"    adata: AnnData object with n_obs x n_vars "
            f"= {self.adata.shape[0]} x {self.adata.shape[1]}"
        )

    @classmethod
    def from_gdrive(
        cls,
        name: str,
        variant: str,
        save_dir: str = os.path.join(f"{get_git_root()}", "datasets"),
    ) -> "PertData":
        """
        Load perturbation dataset from Google Drive.

        Args:
            name: The name of the dataset to load (supported: "dixit", "adamson",
                "norman", "replogle_k562_essential", "replogle_rpe1_essential").
            variant: The variant of the dataset to load (supported: "gears",
                "processed").
            save_dir: The directory to save the data.

        Returns:
            The PertData object.
        """
        # Check if the dataset name is supported
        supported_names = [
            "dixit",
            "adamson",
            "norman",
            "replogle_k562_essential",
            "replogle_rpe1_essential",
        ]
        if name not in supported_names:
            raise ValueError(f"Unknown dataset: {name}")

        # Check if the variant is supported
        supported_variants = ["gears", "processed"]
        if variant not in supported_variants:
            raise ValueError(f"Unknown variant: {variant}")

        # Initialize the PertData object
        instance = cls()
        instance.name = name
        instance.variant = variant
        instance.path = os.path.join(save_dir, name, variant)
        instance.adata = _load(
            dataset_name=name,
            dataset_variant=variant,
            dataset_dir=instance.path,
        )

        # Generate fixed perturbation labels
        instance.adata.obs["condition_fixed"] = _generate_fixed_perturbation_labels(
            labels=instance.adata.obs["condition"]
        )

        return instance

    def normalize_(self, type: str = "CPM") -> None:
        """
        Normalize the perturbation data.

        Args:
            type: The type of normalization to apply (supported: "CPM").
        """
        # Check if the normalization type is supported
        supported_types = ["CPM"]
        if type not in supported_types:
            raise ValueError(f"Unknown normalization type: {type}")

        # Normalize the data
        if type == "CPM":
            _normalize_cpm_(adata=self.adata)
        else:
            raise ValueError(f"Unknown normalization type: {type}")

    def apply_gears_normalization_and_filtering_(self) -> None:
        """Normalize the perturbation data as in GEARS."""
        sc.pp.normalize_total(adata=self.adata, inplace=True)
        sc.pp.log1p(adata=self.adata, inplace=True)
        sc.pp.highly_variable_genes(
            adata=self.adata, n_top_genes=5000, subset=True, inplace=True
        )

    def export_obs_to_csv(self, path: str, obs: List[str]) -> None:
        """
        Export the observation data to a CSV file.

        Args:
            path: The path to save the CSV file.
            obs: The list of observations to export.
        """
        # Get a copy of the observation data
        all_obs = self.adata.obs.copy()

        # Handle special case: "cell_id" is the obs index. Include it as the first
        # column of our temporary DataFrame to be able to export it.
        if "cell_id" in obs:
            all_obs["cell_id"] = self.adata.obs.index
            all_obs = all_obs[
                ["cell_id"] + [item for item in obs if item != "cell_id"]
            ]  # Reorder columns

        # Check if the requested observations are present in the dataset
        for o in obs:
            if o not in all_obs.columns:
                raise ValueError(f"Observation '{o}' not found in the dataset")

        # Make a DataFrame with only the requested observations
        requested_obs = all_obs[obs]

        # Export the observation data to a CSV file
        requested_obs[obs].to_csv(path_or_buf=path, index=False)

    def export_tsv(self, path: str, n_samples: int = None) -> None:
        """
        Save the perturbation data to a TSV file.

        If n_samples is provided, only the first n_samples samples are exported.

        The TSV file has the following format:
        - The first row contains the cell identifiers.
        - The first column contains the gene identifiers.
        - The remaining entries are the gene expression values.

        Args:
            path: The path to save the TSV file.
            n_samples: The number of samples to export.
        """
        # Export all samples if n_samples is not provided
        n_obs = self.adata.shape[0]
        if n_samples is None:
            n_samples = n_obs
            print(f"Exporting all {n_obs} samples to: {path}")
        elif n_samples > n_obs:
            raise ValueError(f"n_samples exceeds available samples. Max is {n_obs}.")
        else:
            print(f"Exporting the first {n_samples}/{n_obs} samples to: {path}")

        # Extract cell identifiers and gene identifiers
        cell_ids = self.adata.obs_names[:n_samples].tolist()
        gene_ids = self.adata.var_names.tolist()

        # Get the first n_samples from the expression matrix
        expression_matrix = self.adata.X[:n_samples, :].todense()

        # Transpose expression matrix to match the desired output (genes as rows,
        # cells as columns)
        expression_matrix = expression_matrix.T

        # Create a DataFrame for export
        expression_df = pd.DataFrame(
            data=expression_matrix, index=gene_ids, columns=cell_ids
        )

        # Reset index to move the gene identifiers (row index) to a column
        expression_df.reset_index(inplace=True)

        # Rename the index column to "Gene" for clarity
        expression_df.rename(columns={"index": "Gene"}, inplace=True)

        # Save the DataFrame to a TSV file
        expression_df.to_csv(path_or_buf=path, sep="\t", index=False)

        print(f"{n_samples} samples exported to: {path}")


def _normalize_cpm_(adata: AnnData) -> None:
    """Normalize the data (target_sum=1e6 is CPM normalization)."""
    sc.pp.normalize_total(adata=adata, target_sum=1e6, inplace=True)


def _load(dataset_name: str, dataset_variant: str, dataset_dir: str) -> AnnData:
    """
    Load perturbation dataset.

    Args:
        dataset_name: The name of the dataset to load (supported: "dixit", "adamson",
            "norman", "replogle_k562_essential", "replogle_rpe1_essential").
        dataset_variant: The variant of the dataset to load (supported: "gears",
            "processed").
        dataset_dir: The directory to save the dataset.

    Returns:
        The perturbation data object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset_name == "dixit":
        if dataset_variant == "gears":
            # url = "https://dataverse.harvard.edu/api/access/datafile/6154416"  # GEARS
            url = "https://drive.google.com/uc?id=1rFgFvEfYeTajMgAB4kHYcU0pI1tQeCSj"
        elif dataset_variant == "processed":
            url = "TBD"
        else:
            raise ValueError(f"Unknown variant: {dataset_variant}")
    elif dataset_name == "adamson":
        if dataset_variant == "gears":
            # url = "https://dataverse.harvard.edu/api/access/datafile/6154417"  # GEARS
            url = "https://drive.google.com/uc?id=1Tnb83H0JAlNAwdsZG40QXIbZjTtCjNT3"
        elif dataset_variant == "processed":
            url = "TBD"
        else:
            raise ValueError(f"Unknown variant: {dataset_variant}")
    elif dataset_name == "norman":
        if dataset_variant == "gears":
            # url = "https://dataverse.harvard.edu/api/access/datafile/6154020"  # GEARS
            url = "https://drive.google.com/uc?id=1cf-esU4ZP5NDbzts1FdVvU5yZeTy4Vmp"
        elif dataset_variant == "processed":
            url = "TBD"
        else:
            raise ValueError(f"Unknown variant: {dataset_variant}")
    elif dataset_name == "replogle_k562_essential":
        if dataset_variant == "gears":
            # url = "https://dataverse.harvard.edu/api/access/datafile/7458695"  # GEARS
            url = "https://drive.google.com/uc?id=1cbl24KIjURm0v7qpQhYFgDf5gXFHSKO5"
        elif dataset_variant == "processed":
            url = "TBD"
        else:
            raise ValueError(f"Unknown variant: {dataset_variant}")
    elif dataset_name == "replogle_rpe1_essential":
        if dataset_variant == "gears":
            # url = "https://dataverse.harvard.edu/api/access/datafile/7458694"  # GEARS
            url = "https://drive.google.com/uc?id=1bNIKNL-a-B6Hj7lcf06GJjPAjNc1WD9a"
        elif dataset_variant == "processed":
            url = "TBD"
        else:
            raise ValueError(f"Unknown variant: {dataset_variant}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    zip_filename = os.path.join(dataset_dir, f"{dataset_name}.zip")
    h5ad_filename = os.path.join(dataset_dir, dataset_name, "perturb_processed.h5ad")

    # If the dataset directory does not exist, create it, download the dataset, and
    # extract it
    if not os.path.exists(path=dataset_dir):
        # Create dataset directory
        print(f"Creating dataset directory: {dataset_dir}")
        os.makedirs(name=dataset_dir)

        # Download the dataset
        print(f"Downloading dataset: {dataset_name}")
        if url.startswith("https://drive.google.com/"):
            print("Downloading from Google Drive")
            gdown.download(url=url, output=zip_filename, quiet=False)
        else:
            print("Downloading from URL")
            download_file(url=url, path=zip_filename)

        # Extract the dataset
        print(f"Extracting dataset: {dataset_name}")
        extract_zip(zip_path=zip_filename, extract_dir=dataset_dir)
    else:
        # print(f"Dataset directory already exists: {dataset_dir}")
        pass

    # Load the dataset
    print(f"Loading dataset: {dataset_name}::{dataset_variant}")
    adata = sc.read_h5ad(filename=h5ad_filename)

    return adata


def _generate_fixed_perturbation_labels(labels: pd.Series) -> pd.Series:
    """
    Generate fixed perturbation labels.

    In the perturbation datasets, single-gene perturbations are expressed as:
    - ctrl+<gene1>
    - <gene1>+ctrl

    Double-gene perturbations are expressed as:
    - <gene1>+<gene2>

    However, in general, there could also be multi-gene perturbations, and they
    might be expressed as a string with additional superfluous "ctrl+" in the
    middle:
        - ctrl+<gene1>+ctrl+<gene2>+ctrl+<gene3>+ctrl

    Hence, we need to remove superfluous "ctrl+" and "+ctrl" matches, such that
    perturbations are expressed as:
    - <gene1> (single-gene perturbation)
    - <gene1>+<gene2> (double-gene perturbation)
    - <gene1>+<gene2>+...+<geneN> (multi-gene perturbation)

    Note: Control cells are not perturbed and are labeled as "ctrl". We do not
    modify these labels.

    Args:
        labels: The perturbation labels.

    Returns:
        The fixed perturbation labels.
    """
    # Remove "ctrl+" and "+ctrl" matches
    labels_fixed = labels.str.replace(pat="ctrl+", repl="")
    labels_fixed = labels_fixed.str.replace(pat="+ctrl", repl="")

    return labels_fixed
