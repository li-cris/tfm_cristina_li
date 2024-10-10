"""Extract the gene epxression from the datasets."""

import os
from os.path import join, exists, dirname
import argparse
from loguru import logger
import scanpy as sc
import pandas as pd
from src.preprocess import consts
    
def run(
    args: argparse.Namespace
) -> None:
    """
    Extract the gene expression from the datasets.

    Notes
    -----
    This function reads an H5AD file, extracts the gene expression data,
    and saves it as a CSV file.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.
        - dataset_name : str
            The name of the dataset to preprocess.
        - gen_geo_dpath : str
            The path to the generated GEO datasets.
        - add_sample_name : bool
            Whether to add the sample name to the output file.
        - out_fpath : str
            The output file path.

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
    
    dataset_name = args.dataset_name
    gen_geo_dpath = args.gen_geo_dpath
    add_sample_name = args.add_sample_name
    
    GEO_ID = consts.DATASET_TO_GEOID[dataset_name]
    
    out_fpath = args.out_fpath or join(
        gen_geo_dpath,
        GEO_ID,
        consts.GENE_EXP_FNAME
    )
    
    logger.info(f"Extracting compass-compatible gene expression from {dataset_name} dataset")
    
    os.makedirs(dirname(out_fpath), exist_ok=True)
    
    h5_fpath = join(
        gen_geo_dpath,
        GEO_ID,
        consts.H5AD_FNAME
    )
    
    logger.debug("Reading H5AD file")
    sc_df = sc.read_h5ad(h5_fpath)
    
    logger.debug("Creating gene expression dataframe")
    gene_exp_df = pd.DataFrame(
        data=sc_df.X.T.A, 
        index=sc_df.var['gene_symbols'],
        columns=sc_df.obs_names if add_sample_name else None
    )
    
    logger.debug(f"Export gene expression as dataframe to {out_fpath}")
    gene_exp_df.to_csv(
        out_fpath,
        header=add_sample_name,
        index=True,
        sep="\t"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data")
    parser.add_argument(
        "--add-sample-name",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--out_fpath",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=consts.AVAIL_DATASETS,
        help="Name of the dataset to preprocess",
    )
    parser.add_argument(
        "gen_geo_dpath",
        type=str,
        help="Path to the generated GEO datasets",
    )
    
    args = parser.parse_args()
    
    run(args)