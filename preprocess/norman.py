import os
import shutil
import tempfile

import gzip
import scanpy as sc

from preprocess.utils import filter_barcodes_and_add_condition
from transmet.utils import download_file, get_git_root


def _modify_features_file(path: str) -> None:
    """
    Modifies the `<prefix>_features.tsv.gz` file in-place to append "Gene Expression".

    Args:
        path: Path to the `<prefix>_features.tsv.gz` file.
    """
    # Read all lines from the file
    with gzip.open(filename=path, mode="rt") as infile:
        lines = infile.readlines()

        if not lines[0].strip().endswith("Gene Expression"):
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file_path = temp_file.name

                with gzip.open(filename=temp_file_path, mode="wt") as outfile:
                    for line in lines:
                        line = line.strip() + "\tGene Expression\n"
                        outfile.write(line)

                shutil.copy2(src=temp_file_path, dst=path)

                print(f"Modified {path} to append 'Gene Expression'")
        else:
            print(
                f"'Gene Expression' already present in {path}, no modification needed"
            )


def _norman_download_raw_data(save_dir: str) -> None:
    """
    Download the raw data for the Norman dataset.

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
        download_file(url=url, path=os.path.join(save_dir, filename))

    # We need to rename "raw_genes.tsv.gz" to "raw_features.tsv.gz", because the
    # function sc.read_10x.mtx() expects the file to be named "raw_features.tsv.gz".
    shutil.move(
        src=os.path.join(save_dir, "raw_genes.tsv.gz"),
        dst=os.path.join(save_dir, "raw_features.tsv.gz"),
    )

    # Also, the "raw_features.tsv.gz" file needs to have a third column with the value
    # "Gene Expression".
    _modify_features_file(path=os.path.join(save_dir, "raw_features.tsv.gz"))


def preprocess(datasets_path: str) -> None:
    """
    Convert the "norman" dataset.

    Args:
        datasets_path: Path to the datasets.
    """
    # All the data is stored in the "datasets" directory. There will be two
    # subdirectories: "raw" and "processed". The "raw" directory will contain the raw
    # data files. The "processed" directory will contain and H5AD file with the raw
    # counts and the normalized counts.
    raw_dir = os.path.join(datasets_path, "norman", "raw")
    processed_dir = os.path.join(datasets_path, "norman", "processed")

    # Create the "processed" directory
    if not os.path.exists(path=processed_dir):
        os.makedirs(name=processed_dir)
    else:
        raise ValueError(
            f"Directory already exists: {processed_dir}. It appears that "
            "the dataset has already been prepared."
        )

    # Create the "raw" directory and download the raw data
    if not os.path.exists(path=raw_dir):
        os.makedirs(name=raw_dir)
        print(f"Downloading the raw data to: {raw_dir}")
        _norman_download_raw_data(save_dir=raw_dir)
    else:
        print(f"Raw data already present in: {raw_dir}")

    # Load the data into an AnnData object
    print(f"Loading the raw data from: {raw_dir}")
    adata = sc.read_10x_mtx(
        path=raw_dir, var_names="gene_ids", cache=False, prefix="raw_"
    )
    print(adata)

    # Filter the data to keep only cells present in the GEARS barcodes file
    print("Filtering the data based on the GEARS barcodes")
    barcodes_filename = os.path.join(get_git_root(), "gears_cell_ids.csv")
    adata_filtered = filter_barcodes_and_add_condition(
        adata=adata, barcodes_filename=barcodes_filename
    )
    print(f"Filtered AnnData to {adata_filtered.shape[0]} cells based on barcodes")
    print(adata_filtered)

    # Save the data to an H5AD file
    # h5ad_dir = os.path.join(processed_dir, "norman")
    h5ad_dir = os.path.join(processed_dir)
    print(f"Saving the processed data to: {h5ad_dir}")
    h5ad_filename = os.path.join(h5ad_dir, "norman.h5ad")
    adata_filtered.write(filename=h5ad_filename)
    print(f"Saved the processed data to: {h5ad_filename}")
