import argparse
import os
import sys

import shutil

import scanpy as sc

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transmet.utils import download_file, get_git_root
from raw_data_converter.utils import (
    filter_barcodes_and_add_condition,
    modify_features_file,
)


def _norman_download_raw_data(save_dir: str) -> None:
    """
    Download the raw data for the "norman" dataset.

    Args:
        save_dir: Directory where the raw data will be saved.
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
        download_file(url=url, save_filename=os.path.join(save_dir, filename))

    # We need to rename "raw_genes.tsv.gz" to "raw_features.tsv.gz", because the
    # function sc.read_10x.mtx() expects the file to be named "raw_features.tsv.gz".
    shutil.move(
        src=os.path.join(save_dir, "raw_genes.tsv.gz"),
        dst=os.path.join(save_dir, "raw_features.tsv.gz"),
    )

    # Also, the "raw_features.tsv.gz" file needs to have a third column with the value
    # "Gene Expression".
    modify_features_file(path=os.path.join(save_dir, "raw_features.tsv.gz"))


def _norman_prepare(datasets_path: str) -> None:
    """
    Prepare the "norman" dataset.

    Args:
        datasets_path: Path to the datasets.
    """
    # All the data is stored in the "datasets" directory. There will be two
    # subdirectories: "raw" and "processed". The "raw" directory will contain the raw
    # data files. The "processed" directory will contain and H5AD file with the raw
    # counts and the normalized counts.
    raw_dir = os.path.join(datasets_path, "norman", "raw")
    processed_dir = os.path.join(datasets_path, "norman", "processed")
    if os.path.exists(path=raw_dir) or os.path.exists(path=processed_dir):
        raise ValueError(
            f"The following directories already exist: '{raw_dir}', '{processed_dir}'. "
            "It appears that the dataset has already been prepared."
        )
    os.makedirs(name=raw_dir)
    os.makedirs(name=processed_dir)

    # Download the raw data
    print(f"Downloading the raw data to: {raw_dir}")
    _norman_download_raw_data(save_dir=raw_dir)

    # Load the data into an AnnData object
    print(f"Loading the raw data from: {raw_dir}")
    adata = sc.read_10x_mtx(
        path=raw_dir, var_names="gene_ids", cache=False, prefix="raw_"
    )
    print(adata)

    # Filter the data to keep only cells present in the GEARS barcodes file
    barcodes_filename = os.path.join(
        get_git_root(), "data", "gears_barcodes", "norman_barcodes.csv"
    )
    adata = filter_barcodes_and_add_condition(
        adata=adata, barcodes_file=barcodes_filename
    )
    print(adata)

    # Save the data to an H5AD file
    h5ad_filename = os.path.join(processed_dir, "norman.h5ad")
    adata.write(filename=os.path.join(processed_dir, "norman.h5ad"))
    print(f"Saved the processed data to: {h5ad_filename}")


def main() -> None:
    """
    Main function.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Raw Data Processor")
    parser.add_argument(
        "dataset_name", type=str, choices=["norman"], help="Dataset name"
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=os.path.join(get_git_root(), "datasets"),
        help="Path to the datasets",
    )
    args = parser.parse_args()

    # Print the arguments
    print(f"Dataset name: {args.dataset_name}")
    print(f"Datasets path: {args.datasets_path}")

    # Do the work
    if args.dataset_name == "norman":
        _norman_prepare(datasets_path=args.datasets_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
