"""Utility functions for preprocessing the data."""

import pandas as pd
from anndata import AnnData


def filter_barcodes_and_add_condition(
    adata: AnnData, barcodes_file_path: str
) -> AnnData:
    """Filter the AnnData object to keep only those cells in the barcodes file.

    Filter the AnnData object to keep only cells present in the barcodes file. Also
    add the "condition_fixed" info for every cell from the barcodes file to the AnnData
    object.

    The barcodes file should have the following format:
    ```
    cell_id,condition_fixed
    AAACATACACCGAT,CREB1
    AAACATACAGAGAT,ctrl
    ...
    ```

    Args:
        adata: The AnnData object to filter.
        barcodes_file_path: The path to the barcodes file.

    Returns:
        The filtered AnnData object.
    """
    # Load the barcodes file.
    barcodes_df = pd.read_csv(filepath_or_buffer=barcodes_file_path, sep=",")

    # Get the barcodes to keep.
    barcodes_to_keep = barcodes_df["cell_id"].values

    # Filter the AnnData object.
    adata_filtered = adata[adata.obs_names.isin(values=barcodes_to_keep)].copy()

    # Add the "condition_fixed" info as "condition".
    barcode_dict = dict(zip(barcodes_df["cell_id"], barcodes_df["condition_fixed"]))
    adata_filtered.obs["condition"] = adata_filtered.obs_names.map(barcode_dict)

    return adata_filtered
