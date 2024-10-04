"""Functionality for handling perturbation datasets."""

import gzip
import os
import shutil
from typing import List, Optional

import gdown
import pandas as pd
import scanpy as sc
from anndata import AnnData


class PertDataset:
    """Class for handling a perturbation dataset.

    The actual perturbation dataset is stored in an
    [AnnData](https://anndata.readthedocs.io/en/latest/) object.

    AnnData is specifically designed for matrix-like data. By this we mean that we have
    n observations, each of which can be represented as d-dimensional vectors, where
    each dimension corresponds to a variable or feature. Both the rows and columns of
    this matrix are special in the sense that they are indexed.

    For instance, in scRNA-seq data:
    - Each row corresponds to a cell with a cell identifier (i.e., barcode).
    - Each column corresponds to a gene with a gene identifier.

    Attributes:
        name: The name of the dataset.
        variant: The variant of the dataset.
        path: The path where the dataset is stored.
        adata: The actual perturbation data.
    """

    def __init__(self, name: str, variant: str, dir_path: str) -> "PertDataset":
        """Initialize the PertDataset object.

        Args:
            name: The name of the dataset.
            variant: The variant of the dataset.
            dir_path: The path to the datasets directory.

        Returns:
            A PertDataset object.
        """
        # Initialize the attributes.
        self.name: Optional[str] = None
        self.variant: Optional[str] = None
        self.path: Optional[str] = None
        self.adata: Optional[AnnData] = None

        # Set the attributes.
        self.name = name
        self.variant = variant
        self.path = os.path.join(dir_path, name, variant)
        self.adata = _load(
            dataset_name=name,
            dataset_variant=variant,
            dataset_dir_path=self.path,
        )

        # Generate fixed perturbation labels.
        self.adata.obs["condition_fixed"] = _generate_condition_fixed(
            labels=self.adata.obs["condition"]
        )

    def __str__(self) -> str:
        """Return a string representation of the PertDataset object."""
        return (
            f"PertDataset object\n"
            f"    name: {self.name}\n"
            f"    variant: {self.variant}\n"
            f"    path: {self.path}\n"
            f"    adata: AnnData object with n_obs x n_vars "
            f"= {self.adata.shape[0]} x {self.adata.shape[1]}"
        )

    def normalize_(self, type: str = "CPM") -> None:
        """Normalize the gene expression matrix.

        Args:
            type: The type of normalization to apply (supported: "CPM").
        """
        if type == "CPM":
            _normalize_cpm_(adata=self.adata)
        else:
            raise ValueError(f"Unsupported normalization type: {type}")

    def export_obs_to_csv(self, file_path: str, obs: List[str]) -> None:
        """Export the observation data to a CSV file.

        Args:
            file_path: The path to save the CSV file.
            obs: The list of observations to export.
        """
        # Get a copy of the observation data.
        all_obs = self.adata.obs.copy()

        # Handle special case: "cell_id" is the obs index. Include it as the first
        # column of our temporary DataFrame to be able to export it.
        if "cell_id" in obs:
            all_obs["cell_id"] = self.adata.obs.index
            all_obs = all_obs[
                ["cell_id"] + [item for item in obs if item != "cell_id"]
            ]  # Reorder columns.

        # Check if the requested observations are present in the dataset.
        for o in obs:
            if o not in all_obs.columns:
                raise ValueError(f"Observation '{o}' not found in the dataset.")

        # Make a DataFrame with only the requested observations.
        requested_obs = all_obs[obs]

        # Export the observation data to a CSV file.
        requested_obs[obs].to_csv(path_or_buf=file_path, index=False)


def _load(dataset_name: str, dataset_variant: str, dataset_dir_path: str) -> AnnData:
    """Load perturbation dataset.

    Args:
        dataset_name: The name of the dataset.
        dataset_variant: The variant of the dataset.
        dataset_dir_path: The directory path where the dataset is stored.

    Returns:
        The perturbation dataset as an AnnData object.

    Raises:
        ValueError: If the dataset name or dataset variant is unsupported.
    """
    if dataset_name == "adamson":
        if dataset_variant == "gears":
            url = "https://drive.google.com/uc?id=1W1phErDoQ9U5iJZSEuyEZM4dF8U8ZWqf"
        elif dataset_variant == "preprocessed":
            url = "TBD"
        else:
            raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    elif dataset_name == "dixit":
        if dataset_variant == "gears":
            url = "https://drive.google.com/uc?id=1BN6gwKFgJIpR9fXfdQ9QeHm8mAzvmhKQ"
        elif dataset_variant == "preprocessed":
            url = "TBD"
        else:
            raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    elif dataset_name == "norman":
        if dataset_variant == "gears":
            url = "https://drive.google.com/uc?id=1T5_varFOGWUtSig4RQRCSsfPxivUwd9j"
        elif dataset_variant == "preprocessed":
            url = "TBD"
        else:
            raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    elif dataset_name == "replogle_k562_essential":
        if dataset_variant == "gears":
            url = "https://drive.google.com/uc?id=12flxmpj-XnJ8BZKtf-sgBhdUN2X4v7CD"
        elif dataset_variant == "preprocessed":
            url = "TBD"
        else:
            raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    elif dataset_name == "replogle_rpe1_essential":
        if dataset_variant == "gears":
            url = "https://drive.google.com/uc?id=1b-ZwE_Y6dNKqb4KQgUgFKfl6OGC8lmYE"
        elif dataset_variant == "preprocessed":
            url = "TBD"
        else:
            raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    # Make the directory for the selected dataset.
    os.makedirs(name=dataset_dir_path, exist_ok=True)

    # Make the file path for the selected dataset.
    h5ad_file_path = os.path.join(dataset_dir_path, "adata.h5ad")

    # Download and extract the dataset if it does not exist.
    if not os.path.exists(path=h5ad_file_path):
        # Make the file path for the compressed dataset.
        zip_file_path = os.path.join(dataset_dir_path, "adata.h5ad.gz")

        # Download the dataset.
        print(f"Downloading: {url} -> {zip_file_path}")
        gdown.download(url=url, output=zip_file_path)

        # Extract the dataset.
        print(f"Extracting: {zip_file_path} -> {h5ad_file_path}")
        with gzip.open(filename=zip_file_path, mode="rb") as f_in:
            with open(file=h5ad_file_path, mode="wb") as f_out:
                shutil.copyfileobj(fsrc=f_in, fdst=f_out)

        # Remove the compressed file.
        print(f"Removing: {zip_file_path}")
        os.remove(path=zip_file_path)

    # Load the dataset.
    print(f"Loading: {h5ad_file_path}")
    adata = sc.read_h5ad(filename=h5ad_file_path)

    return adata


def _generate_condition_fixed(labels: pd.Series) -> pd.Series:
    """Generate fixed perturbation labels.

    In the perturbation datasets, single-gene perturbations are expressed as:
    - ctrl+<gene1>
    - <gene1>+ctrl

    Double-gene perturbations are expressed as:
    - <gene1>+<gene2>

    However, in general, there could also be multi-gene perturbations, and they might be
    expressed as a string with additional superfluous "ctrl+" in the middle:
    - ctrl+<gene1>+ctrl+<gene2>+ctrl+<gene3>+ctrl

    Hence, we need to remove superfluous "ctrl+" and "+ctrl" matches, such that
    perturbations are expressed as:
    - <gene1> (single-gene perturbation)
    - <gene1>+<gene2> (double-gene perturbation)
    - <gene1>+<gene2>+...+<geneN> (multi-gene perturbation)

    Note: Control cells are not perturbed and are labeled as "ctrl". We do not modify
    these labels.

    Args:
        labels: The perturbation labels.

    Returns:
        The fixed perturbation labels.
    """
    # Remove "ctrl+" and "+ctrl" matches.
    labels_fixed = labels.str.replace(pat="ctrl+", repl="")
    labels_fixed = labels_fixed.str.replace(pat="+ctrl", repl="")

    return labels_fixed


def _normalize_cpm_(adata: AnnData) -> None:
    """Normalize the data (target_sum=1e6 is CPM normalization).

    Args:
        adata: The AnnData object to normalize.
    """
    sc.pp.normalize_total(adata=adata, target_sum=1e6, inplace=True)
