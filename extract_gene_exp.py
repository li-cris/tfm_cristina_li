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
        - first_num_samples : int
            Extract only the first n samples.
        - num_samples_per_tsv : int
            Create multiple tsv files, each contains maximum n samples.

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
    first_num_samples = args.first_num_samples
    num_samples_per_tsv = args.num_samples_per_tsv

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
    adata = sc.read_h5ad(h5_fpath)

    logger.debug("Creating gene expression dataframe")
    gene_exp_df = pd.DataFrame(
        data=adata.X.T.A,
        index=adata.var['gene_symbols'],
        columns=adata.obs_names if add_sample_name else None
    )

    if first_num_samples is not None:
        gene_exp_df = gene_exp_df.iloc[:, :first_num_samples]


    if num_samples_per_tsv is None:
        logger.debug(f"Export gene expression as dataframe to {out_fpath}")
        gene_exp_df.to_csv(
            out_fpath,
            header=add_sample_name,
            index=True,
            sep="\t"
        )
    else:
        num_samples = len(gene_exp_df.columns)
        slice_range = range(0, num_samples-num_samples_per_tsv, num_samples_per_tsv)
        for it, start_col_id in enumerate(slice_range):
            tmp_gene_exp_df = gene_exp_df.iloc[:, start_col_id:(start_col_id+num_samples_per_tsv)]

            out_dpath = join(
                gen_geo_dpath,
                GEO_ID,
                "expressions",
            )
            out_fpath = join(out_dpath, f"expression.{it}.tsv")
            
            logger.debug(f"Export gene expression as dataframe to {out_fpath}")

            os.makedirs(out_dpath, exist_ok=True)

            tmp_gene_exp_df.to_csv(
                out_fpath,
                header=add_sample_name,
                index=True,
                sep="\t"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract gene expression from datasets"
    )

    #? Required arguments
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=consts.AVAIL_DATASETS,
        help="Name of the dataset to extract gene expression from",
    )
    parser.add_argument(
        "gen_geo_dpath",
        type=str,
        help="Path to the generated GEO datasets",
    )

    #? Optional arguments
    parser.add_argument(
        "--first-num-samples",
        type=int,
        help="Extract only the first n samples",
    )
    parser.add_argument(
        "--num-samples-per-tsv",
        type=int,
        help="Create multiple tsv files, each contains maximum n samples",
    )
    parser.add_argument(
        "--add-sample-name",
        action="store_true",
        help="Add sample names as header",
    )
    parser.add_argument(
        "--out-fpath",
        type=str,
        help="Output file path",
    )

    args = parser.parse_args()

    run(args)