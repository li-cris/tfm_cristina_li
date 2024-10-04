import pandas as pd
from anndata import AnnData


def filter_barcodes_and_add_condition(
    adata: AnnData, barcodes_filepath: str
) -> AnnData:
    """Filter the AnnData object to keep only those cells in the the barcodes file.

    Filters the AnnData object to keep only cells present in the barcodes file. Also
    adds the "condition" info for every cell from the barcodes file to the AnnData
    object.

    The barcodes file should have the following format:
    ```
    cell_id, condition
    AAACATACACCGAT, CREB1
    AAACATACAGAGAT, ctrl
    ...
    ```

    Args:
        adata: AnnData object containing the gene expression data.
        barcodes_filepath: Path to the barcodes file.

    Returns:
        The filtered AnnData object.
    """
    # Load the barcodes file
    barcodes_df = pd.read_csv(filepath_or_buffer=barcodes_filepath, sep=",")

    # Get the barcodes to keep
    barcodes_to_keep = barcodes_df["cell_id"].values

    # Filter the AnnData object
    adata_filtered = adata[adata.obs_names.isin(values=barcodes_to_keep)].copy()

    # Add the "condition" info
    barcode_dict = dict(zip(barcodes_df["cell_id"], barcodes_df["condition"]))
    adata_filtered.obs["condition"] = adata_filtered.obs_names.map(barcode_dict)

    return adata_filtered
