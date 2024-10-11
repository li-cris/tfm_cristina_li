"""Preprocessing for the Adamson dataset."""

import os
from os.path import join, exists
import tarfile
import gzip
import shutil
import tempfile as tmp
from loguru import logger
import pandas as pd
import anndata as ad
import scanpy as sc
import gdown
from . import consts

DATASET_NAME = "adamson"
GEO_ID = consts.DATASET_TO_GEOID[DATASET_NAME]
REL_DNAME = "suppl"
FNAMES = {
    "barcodes.tsv.gz": "raw_barcodes.tsv.gz",
    "cell_identities.csv.gz": "raw_cell_identities.csv.gz",
    "genes.tsv.gz": "raw_features.tsv.gz",
    "matrix.mtx.txt.gz": "raw_matrix.mtx.gz",
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

    #? Check if the dataset directory exists
    assert exists(dataset_dpath)

    out_dpath = join(
        out_dpath,
        GEO_ID
    )

    logger.debug(f"Creating directory {out_dpath}")
    os.makedirs(out_dpath, exist_ok=True)
    
    tar_fpath = join(
        geo_dpath,
        GEO_ID,
        REL_DNAME,
        f"{GEO_ID}_RAW.tar"
    )
    #? Extract the TAR file.
    logger.debug(f"Extracting files in {tar_fpath}")
    with tarfile.open(tar_fpath, mode="r") as tar_f:        
        tar_fnams_ps = pd.Series(tar_f.getnames())
        tar_fnames_df = tar_fnams_ps.str.split(
            '_', 
            expand=True, 
            n=2
        )
        tar_fnames_df.insert(3, 'fname', tar_fnams_ps)
        tar_fnames_df.rename(
            columns={
                0: 'exp_name', 
                1: 'exp_config',
                2:'new_fname'
            }, 
            inplace=True
        )
        
        for _, rec in tar_fnames_df.iterrows():
            out_exp_dpath = join(
                out_dpath,
                rec['exp_name'],
            )
            os.makedirs(out_exp_dpath, exist_ok=True)
            
            in_fname = rec['fname']
            out_fname = FNAMES[rec['new_fname']]
            
            gz_in_fpath = join(
                out_exp_dpath,
                in_fname
            )
            
            gz_out_fpath = join(
                out_exp_dpath,
                out_fname
            )
            
            logger.debug(f"Extracting {in_fname} to {out_exp_dpath}")
            tar_f.extract(
                in_fname,
                out_exp_dpath
            )
            
            logger.debug(f"Renaming {in_fname} to {out_fname}")
            shutil.move(
                gz_in_fpath,
                gz_out_fpath,
            )
            
            if out_fname == "raw_features.tsv.gz":
                logger.debug(f"Processing feature file {gz_out_fpath}")
                df = pd.read_csv(
                    gz_out_fpath,
                    compression='gzip',
                    header=None,
                    sep="\t",
                )
                df[2] = "Gene Expression"
                logger.debug(f"Storing feature file {gz_out_fpath}")
                df.to_csv(
                    gz_out_fpath,
                    header=False,
                    index=False,
                    compression='infer',
                    sep="\t",
                )
                
    adata_dict = {}
    for exp_name in tar_fnames_df['exp_name'].unique():
        out_exp_dpath = join(
            out_dpath,
            exp_name,
        )
        
        logger.debug(f"Creating 10x-Genomic-formatted array from {out_exp_dpath}")
        adata = sc.read_10x_mtx(
            path=out_exp_dpath,
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
            gears_obs_fpath = join(out_exp_dpath, consts.GEARS_OBS_FNAME)
            req_obs_df.to_csv(
                gears_obs_fpath,
                index=False
            )

            sel_cell_ids = req_obs_df[consts.CELL_ID_COLNAME].values

            adata = adata[adata.obs_names.isin(values=sel_cell_ids.flatten())]

        h5ad_fpath = join(out_exp_dpath, consts.H5AD_FNAME)
        adata.write(h5ad_fpath)
        
        adata_dict[exp_name] = adata
        
    logger.debug("Merging data from all experiments")
    adata = ad.concat(
        adata_dict.values(), 
        axis=0, 
        join="outer", 
        index_unique=None
    )
    
    h5ad_fpath = join(out_dpath, consts.H5AD_FNAME)
    
    logger.debug(f"Writing to {h5ad_fpath}")
    adata.write(h5ad_fpath)