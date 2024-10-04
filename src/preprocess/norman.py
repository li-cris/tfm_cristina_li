"""Preprocess the Norman dataset."""

import gzip
import os
import shutil
import sys
import tempfile

import scanpy as sc

# Add the root of the project to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.extract_gears_obs import extract_gears_obs
from preprocess.utils import filter_barcodes_and_add_condition
from transmet.utils import download_file


def _modify_features_file(path: str) -> None:
    """Modify the `<prefix>_features.tsv.gz` file in-place to append "Gene Expression".

    Args:
        path: The path to the `<prefix>_features.tsv.gz` file.
    """
    with gzip.open(filename=path, mode="rt") as input_file:
        lines = input_file.readlines()
        if not lines[0].strip().endswith("Gene Expression"):
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file_path = temp_file.name
                with gzip.open(filename=temp_file_path, mode="wt") as output_file:
                    for line in lines:
                        line = line.strip() + "\tGene Expression\n"
                        output_file.write(line)
                shutil.copy2(src=temp_file_path, dst=path)


def _download_raw_data(dir_path: str) -> None:
    """Download the raw data.

    Args:
        dir_path: The directory path where the raw data will be stored.
    """
    urls = [
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE133344&format=file&file=GSE133344%5Fraw%5Fbarcodes%2Etsv%2Egz",
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE133344&format=file&file=GSE133344%5Fraw%5Fcell%5Fidentities%2Ecsv%2Egz",
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE133344&format=file&file=GSE133344%5Fraw%5Fgenes%2Etsv%2Egz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344%5Fraw%5Fmatrix%2Emtx%2Egz",
    ]

    filenames = [
        "raw_barcodes.tsv.gz",
        "raw_cell_identities.csv.gz",
        "raw_genes.tsv.gz",
        "raw_matrix.mtx.gz",
    ]

    for url, filename in zip(urls, filenames):
        download_file(url=url, path=os.path.join(dir_path, filename))

    # We need to rename "raw_genes.tsv.gz" to "raw_features.tsv.gz", because the
    # function sc.read_10x_mtx() expects the file to be named "raw_features.tsv.gz".
    # We make a copy and keep "raw_genes.tsv.gz" to avoid duplicate downloads.
    shutil.copy2(
        src=os.path.join(dir_path, "raw_genes.tsv.gz"),
        dst=os.path.join(dir_path, "raw_features.tsv.gz"),
    )

    # Also, the "raw_features.tsv.gz" file needs to have a third column with the value
    # "Gene Expression".
    _modify_features_file(path=os.path.join(dir_path, "raw_features.tsv.gz"))


def preprocess(datasets_dir_path: str, apply_gears_filter: bool = False) -> None:
    """Preprocess the Norman dataset.

    Args:
        datasets_dir_path: The path to the datasets directory.
        apply_gears_filter: Whether to reduce the data to the same set of cells as used
            by GEARS.
    """
    # Create the "raw" directory and download the raw data.
    raw_dir_path = os.path.join(datasets_dir_path, "norman", "raw")
    os.makedirs(name=raw_dir_path, exist_ok=True)
    _download_raw_data(dir_path=raw_dir_path)

    # Load the data into an AnnData object.
    print(f"Loading raw data from: {raw_dir_path}")
    adata = sc.read_10x_mtx(
        path=raw_dir_path, var_names="gene_ids", cache=False, prefix="raw_"
    )
    print(adata)

    if apply_gears_filter:
        # Extract the GEARS barcodes.
        gears_barcodes_file_path = extract_gears_obs(
            dataset_name="norman", datasets_dir_path=datasets_dir_path
        )

        # Filter the data to keep only those cells as used by GEARS.
        print(f"Filtering the data based on: {gears_barcodes_file_path}")
        adata = filter_barcodes_and_add_condition(
            adata=adata, barcodes_file_path=gears_barcodes_file_path
        )
        print(f"Remaining cells: {adata.shape[0]}")
        print(adata)

    # Create the "preprocessed" directory.
    preprocessed_dir_path = os.path.join(datasets_dir_path, "norman", "preprocessed")
    os.makedirs(name=preprocessed_dir_path, exist_ok=True)

    # Save the data to an H5AD file.
    h5ad_file_path = os.path.join(preprocessed_dir_path, "adata.h5ad")
    print(f"Saving the processed data to: {h5ad_file_path}")
    adata.write(filename=h5ad_file_path)
