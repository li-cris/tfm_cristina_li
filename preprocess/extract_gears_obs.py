"""Utility to extract ['cell_barcode', 'condition', 'cell_type'] from GEARS datasets."""

import argparse
import os
import sys

import pandas as pd

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transmet.utils import get_git_root
from transmet import pertdata as pt


def _fix_adamson_csv(path: str) -> None:
    """Fix the CSV file for the Adamson dataset."""
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath_or_buffer=path)

    # Remove the "(?)" suffix from all rows
    print(df.head())
    df = df.replace(to_replace=r"\(\?\)", value="", regex=True)
    print(df.head())

    # Save the modified DataFrame back to the same CSV file
    df.to_csv(path_or_buf=path, index=False)


def main() -> None:
    """
    Main function.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Small utility to extract observations from the GEARS datasets"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=[
            "dixit",
            "adamson",
            "norman",
            "replogle_k562_essential",
            "replogle_rpe1_essential",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=os.path.join(get_git_root(), "datasets"),
        help="Path to the datasets directory",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file",
    )
    args = parser.parse_args()

    # Export the data
    pert_data = pt.PertData.from_gdrive(name=args.dataset_name, variant="gears")
    print(f"Exporting data to: {args.output_file}")
    pert_data.export_obs_to_csv(path=args.output_file, obs=["cell_id", "condition"])

    # Apply fixes
    if args.dataset_name == "adamson":
        _fix_adamson_csv(filepath=args.output_file)


if __name__ == "__main__":
    main()
