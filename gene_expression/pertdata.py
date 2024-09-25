"""Functionality for handling perturbation data."""

import os
from typing import Optional

from anndata import AnnData
import pandas as pd
import scanpy as sc

from utils import download_file, extract_zip


class PertData:
    """
    Class for perturbation data.

    The perturbation data is stored in an `AnnData` object. Refer to
    https://anndata.readthedocs.io/en/latest/ for more information on `AnnData`.

    `AnnData` is specifically designed for matrix-like data. By this we mean that we
    have n observations, each of which can be represented as d-dimensional vectors,
    where each dimension corresponds to a variable or feature. Both the rows and columns
    of this matrix are special in the sense that they are indexed.

    For instance, in scRNA-seq data, each row corresponds to a cell with a barcode, and
    each column corresponds to a gene with a gene identifier.

    Attributes:
        name: The name of the dataset.
        path: The path where the dataset is stored.
        adata: The actual perturbation data.
    """

    def __init__(self) -> None:
        """Initialize the PertData object."""
        self.name: Optional[str] = None
        self.path: Optional[str] = None
        self.adata: Optional[AnnData] = None

    def __str__(self) -> str:
        """Return a string representation of the PertData object."""
        return (
            f"PertData object\n"
            f"    name: {self.name}\n"
            f"    path: {self.path}\n"
            f"    adata: AnnData object with n_obs x n_vars = {self.adata.shape[0]} x {self.adata.shape[1]}"
        )

    @classmethod
    def from_repo(cls, name: str, save_dir: str) -> "PertData":
        """
        Load perturbation dataset from an online repository.

        Args:
            name: The name of the dataset to load (supported: "dixit", "adamson",
                "norman").
            save_dir: The directory to save the data.
        """
        instance = cls()
        instance.name = name
        instance.path = os.path.join(save_dir, instance.name)
        instance.adata = _load(dataset_name=name, dataset_dir=instance.path)
        instance.adata.obs["condition_fixed"] = generate_fixed_perturbation_labels(
            labels=instance.adata.obs["condition"]
        )
        return instance


def _load(dataset_name: str, dataset_dir: str) -> AnnData:
    """
    Load perturbation dataset.

    The following are the [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/)
    accession numbers used:
    - Dixit et al., 2016: [GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063)
    - Adamson et al., 2016: [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)
    - Norman et al., 2019: [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)

    The following are the DOIs of the corresponding publications:
    - Dixit et al., 2016: https://doi.org/10.1016/j.cell.2016.11.038
    - Adamson et al., 2016: https://doi.org/10.1016/j.cell.2016.11.048
    - Norman et al., 2019: https://doi.org/10.1126/science.aax4438

    The following URLs are from the
    [GEARS code](https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py):
    - Dixit et al., 2016: https://dataverse.harvard.edu/api/access/datafile/6154416
    - Adamson et al., 2016: https://dataverse.harvard.edu/api/access/datafile/6154417
    - Norman et al., 2019: https://dataverse.harvard.edu/api/access/datafile/6154020

    Args:
        dataset_name: The name of the dataset to load (supported: "dixit", "adamson",
            "norman").
        dataset_dir: The directory to save the dataset.

    Returns:
        The perturbation data object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset_name == "dixit":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154416"
    elif dataset_name == "adamson":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154417"
    elif dataset_name == "norman":
        url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
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
        download_file(url=url, save_filename=zip_filename)

        # Extract the dataset
        print(f"Extracting dataset: {dataset_name}")
        extract_zip(zip_path=zip_filename, extract_dir=dataset_dir)
    else:
        print(f"Dataset directory already exists: {dataset_dir}")

    # Load the dataset
    print(f"Loading dataset: {dataset_name}")
    adata = sc.read_h5ad(filename=h5ad_filename)

    return adata


def generate_fixed_perturbation_labels(labels: pd.Series) -> pd.Series:
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
