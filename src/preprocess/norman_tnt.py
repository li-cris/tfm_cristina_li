"""Preprocessing for the Norman dataset."""

import os
from os.path import join, exists
import gzip
import shutil
import tempfile as tmp
from loguru import logger
import pandas as pd
import scanpy as sc
import gdown
from . import consts

DATASET_NAME = "norman"
GEO_ID = consts.DATASET_TO_GEOID[DATASET_NAME]
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
    copy: bool = False,
    filter_by_gears: bool = False,
) -> None:
    """
    Prepares raw data for further processing by creating output directories,
    processing feature files, and creating symbolic links to input files.

    Notes
    -----
    This function assumes that the input directory path and output directory path
    are valid and that the dataset directory exists.

    Parameters
    ----------
    geo_dpath : str
        The path to the GEO dataset directory.
    out_dpath : str
        The path to the output directory.
    copy : bool, optional
        Whether to copy the input files instead of creating symbolic links.
        Defaults to False.
    filter_by_gears : bool, optional
        Whether to filter the data using the GEARS dataset. Defaults to False.

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

    #? Create the output directory path
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
                index=False,
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
                if copy:                   
                    try:
                        shutil.copy2(
                            src=in_fpath,
                            dst=out_fpath,
                        )
                    except FileExistsError:
                        os.remove(out_fpath)
                        shutil.copy2(
                            src=in_fpath,
                            dst=out_fpath,
                        )
                else:
                    os.symlink(
                        in_fpath,
                        out_fpath
                    )
            except FileExistsError:
                pass

    logger.debug(f"Creating 10x-Genomic-formatted array from {out_dpath}")
    adata = sc.read_10x_mtx(
        path=out_dpath,
        var_names="gene_ids",
        cache=False,
        prefix="raw_"
    )

    if filter_by_gears:
        gears_ds_url = consts.DATASET_TO_GDRIVE_URL[DATASET_NAME]

        logger.debug(f"Downloading gears dataset from {gears_ds_url}")
        with tmp.TemporaryDirectory() as tmp_d:
            gears_gz_fpath = join(tmp_d, consts.GZ_FNAME)
            gears_h5_fpath = join(tmp_d, consts.H5AD_FNAME)
            with open(gears_gz_fpath, 'wb') as gz_f:
                gdown.download(
                    url=gears_ds_url,
                    output=gz_f
                )

            #? Extract GZIP file
            with gzip.open(gears_gz_fpath, mode="rb") as f_in:
                with open(gears_h5_fpath, mode="wb") as f_out:
                    shutil.copyfileobj(fsrc=f_in, fdst=f_out)

            gears_adata = sc.read_h5ad(filename=gears_h5_fpath)

        all_obs_df = gears_adata.obs.copy()

        #? move cell_id from index to a column
        obs_cols = [consts.CELL_ID_COLNAME, "condition"]
        if consts.CELL_ID_COLNAME in obs_cols:
            all_obs_df.insert(
                0, 
                consts.CELL_ID_COLNAME, 
                gears_adata.obs.index
            )

        #? Check if the requested observations are present in the dataset.
        for o in obs_cols:
            if o not in all_obs_df.columns:
                raise ValueError(f"Observation '{o}' not found in the dataset.")

        #? Filter columns
        req_obs_df = all_obs_df.loc[:, obs_cols]

        #? Export the observation data to a CSV file.
        gears_obs_fpath = join(out_dpath, consts.GEARS_OBS_FNAME)
        req_obs_df.to_csv(
            gears_obs_fpath,
            index=False
        )

        sel_cell_ids = req_obs_df[consts.CELL_ID_COLNAME,].values

        adata = adata[adata.obs_names.isin(values=sel_cell_ids)]

    out_fpath = join(out_dpath, consts.H5AD_FNAME)
    adata.write(out_fpath)