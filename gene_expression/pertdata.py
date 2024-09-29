"""Functionality for handling perturbation data."""

import os
from typing import Optional

from anndata import AnnData
import gdown
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
                "norman", "replogle_k562_essential", "replogle_rpe1_essential").
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

    The following are the DOIs of the corresponding publications:
    - Dixit et al., 2016: https://doi.org/10.1016/j.cell.2016.11.038
    - Adamson et al., 2016: https://doi.org/10.1016/j.cell.2016.11.048
    - Norman et al., 2019: https://doi.org/10.1126/science.aax4438
    - Replogle et al., 2022: https://doi.org/10.1016/j.cell.2022.05.013

    Args:
        dataset_name: The name of the dataset to load (supported: "dixit", "adamson",
            "norman", "replogle_k562_essential", "replogle_rpe1_essential").
        dataset_dir: The directory to save the dataset.

    Returns:
        The perturbation data object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset_name == "dixit":
        # Copy on Harvard Dataverse (used in the GEARS code)
        # url = "https://dataverse.harvard.edu/api/access/datafile/6154416"
        # Copy on Jan's UNAV Google Drive
        url = "https://drive.google.com/uc?id=1rFgFvEfYeTajMgAB4kHYcU0pI1tQeCSj"
    elif dataset_name == "adamson":
        # Copy on Harvard Dataverse (used in the GEARS code)
        # url = "https://dataverse.harvard.edu/api/access/datafile/6154417"
        # Copy on Jan's UNAV Google Drive
        url = "https://drive.google.com/uc?id=1Tnb83H0JAlNAwdsZG40QXIbZjTtCjNT3"
    elif dataset_name == "norman":
        # Copy on Harvard Dataverse (used in the GEARS code)
        # url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
        # Copy on Jan's UNAV Google Drive
        url = "https://drive.google.com/uc?id=1cf-esU4ZP5NDbzts1FdVvU5yZeTy4Vmp"
    elif dataset_name == "replogle_k562_essential":
        # Copy on Harvard Dataverse (used in the GEARS code)
        # url = "https://dataverse.harvard.edu/api/access/datafile/7458695"
        # Copy on Jan's UNAV Google Drive
        url = "https://drive.google.com/uc?id=1cbl24KIjURm0v7qpQhYFgDf5gXFHSKO5"
    elif dataset_name == "replogle_rpe1_essential":
        # Copy on Harvard Dataverse (used in the GEARS code)
        # url = "https://dataverse.harvard.edu/api/access/datafile/7458694"
        # Copy on Jan's UNAV Google Drive
        url = "https://drive.google.com/uc?id=1bNIKNL-a-B6Hj7lcf06GJjPAjNc1WD9a"
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
            print("Downloading from Jan's UNAV Google Drive")
            gdown.download(url=url, output=zip_filename, quiet=False)
        else:
            print("Downloading from Havard Dataverse")
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
