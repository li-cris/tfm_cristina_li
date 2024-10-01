import os

import gzip
import pandas as pd


def modify_features_file(path):
    """Modifies the features file to append 'Gene Expression' if not already present."""
    temp_file_path = "temp_genes.tsv.gz"

    with gzip.open(path, "rt") as f_in:
        lines = f_in.readlines()

    if not lines[0].strip().endswith("Gene Expression"):
        with gzip.open(temp_file_path, "wt") as f_out:
            for line in lines:
                line = line.strip() + "\tGene Expression\n"
                f_out.write(line)
        os.replace(temp_file_path, path)
        print(f"Modified {path} to add 'Gene Expression'.")
    else:
        print(f"'Gene Expression' already present in {path}, no modification needed.")


def filter_barcodes_and_add_condition(adata, barcodes_file):
    """
    Filters the AnnData object to keep only cells present in the barcode-to-cell-type mapping file and adds condition info.

    Args:
    - adata (AnnData): AnnData object containing the gene expression data.
    - barcode_file (str): Path to the barcode file containing cell barcodes and their corresponding conditions.
    """
    barcode_df = pd.read_csv(barcodes_file, sep=",")
    barcodes_to_keep = barcode_df["cell_barcode"].values
    adata = adata[adata.obs_names.isin(barcodes_to_keep)].copy()

    barcode_dict = dict(zip(barcode_df["cell_barcode"], barcode_df["condition"]))
    adata.obs["condition"] = adata.obs_names.map(barcode_dict)

    print(f"Filtered AnnData to {adata.shape[0]} cells based on barcodes.")
    return adata
