"""Preprocessing for the Norman dataset."""

import os
import sys
from os.path import join, exists
from loguru import logger
import gzip

import pandas as pd
import scanpy as sc

GEO_ID = "GSE133344"
REL_DNAME = "suppl"
FNAMES = [
    "raw_barcodes.tsv.gz",
    "raw_cell_identities.csv.gz",
    "raw_genes.tsv.gz",
    "raw_matrix.mtx.gz",
]

FEATS_FNAMES = {
    "raw_genes.tsv.gz": "raw_features.tsv.gz"
}

def prepare_raw_data(
    geo_dpath: str,
    out_dpath: str,
) -> None:
    """
    Prepares the raw data for processing by creating the necessary directory structure and symbolic links.

    Notes
    -----
    This function assumes that the input directory structure is as follows:
    - `geo_dpath` contains a subdirectory named `GEO_ID`
    - `GEO_ID` contains a subdirectory named `REL_DNAME`
    - `REL_DNAME` contains files with names matching the patterns in `FNAMES`

    Parameters
    ----------
    geo_dpath : str
        The path to the input directory containing the GEO data.
    out_dpath : str
        The path to the output directory where the prepared data will be written.

    Returns
    -------
    None

    Examples
    --------

    Authors
    -------
    - Yeremia G. Adhisantoso (adhisant@tnt.uni-hannover.de)
    - Llama3.1 70B - 4.0bpw
    """

    #? Create the output directory path by joining the output directory path with the GEO ID
    dataset_dpath = join(
        geo_dpath,
        GEO_ID,
    )

    out_dpath = join(
        out_dpath,
        GEO_ID
    )

    logger.debug(f"Creating directory {out_dpath}")
    os.makedirs(out_dpath, exist_ok=True)

    #? Check if the dataset directory exists
    assert exists(dataset_dpath)

    #? Iterate over the file names
    for in_fname in FNAMES:
        #? Construct the input file path
        in_fpath = join(
            dataset_dpath,
            REL_DNAME,
            f"{GEO_ID}_{in_fname}"
        )

        #? Check if the input file exists
        assert exists(in_fpath)

        #? Check if the file is a feature file
        if in_fname in FEATS_FNAMES:
            #? Construct the output file name
            out_fname = FEATS_FNAMES[in_fname]
            out_fpath = join(
                out_dpath,
                out_fname
            )

            #? Process the feature file
            logger.debug(f"Processing feature file {in_fpath}")           
            df = pd.read_csv(
                in_fpath,
                compression='gzip',
                header=None,
                sep="\t",
            )
            df[2] = "Gene Expression"
            logger.debug(f"Storing feature file {out_fpath}")
            df.to_csv(
                out_fpath,
                header=False,
                compression='infer',
                sep="\t",
            )
        else:
            #? Construct the output file name
            out_fname = in_fname
            out_fpath = join(
                out_dpath,
                out_fname
            )

            #? Create a symbolic link to the input file to avoid copying
            try:
                logger.debug(f"Creating symbolic link {out_fpath}")
                os.symlink(in_fpath, out_fpath)
            except FileExistsError:
                pass
            
    logger.debug(f"Creating 10x-Genomic-formatted array from {out_dpath}")
    adata = sc.read_10x_mtx(
        path=out_dpath, 
        var_names="gene_ids", 
        cache=False, 
        prefix="raw_"
    )
    return adata