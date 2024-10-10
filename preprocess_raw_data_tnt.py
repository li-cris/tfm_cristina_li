"""Preprocess the datasets."""

import os
import argparse
from loguru import logger

from src.preprocess import consts
from src.preprocess import norman_tnt
    
def run(args):
    dataset_name = args.dataset_name
    geo_dpath = args.geo_dpath
    out_dpath = args.out_dpath
    
    if dataset_name == 'norman':
        norman_tnt.prepare_raw_data(
            geo_dpath, 
            out_dpath,
            copy=args.copy,
            filter_by_gears=args.filter_by_gears
        )
    else:
        raise ValueError(f"Invalid datasets: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the raw data instead of creating symbolic link",
    )
    parser.add_argument(
        "--filter-by-gears",
        action="store_true",
        help="Filter the data based on GEARS dataset",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=consts.AVAIL_DATASETS,
        help="Name of the dataset to preprocess",
    )
    parser.add_argument(
        "geo_dpath",
        type=str,
        help="Path to folder where GEO datasets are stored",
    )
    parser.add_argument(
        "out_dpath",
        type=str,
        help="Path to folder where GEO datasets are stored",
    )
    
    args = parser.parse_args()
    
    run(args)