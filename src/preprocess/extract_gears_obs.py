"""Utility to extract ['cell_id', 'condition_fixed'] from the GEARS datasets."""

import os
import sys

import pandas as pd

# Add the root of the project to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transmet.pert_dataset as pt


def _fix_adamson(csv_file_path: str) -> None:
    """Fix the CSV file for the Adamson dataset.

    Args:
        csv_file_path: The path to the CSV file.
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(filepath_or_buffer=csv_file_path)

    # Remove the "(?)" suffix from all rows.
    df = df.replace(to_replace=r"\(\?\)", value="", regex=True)

    # Save the modified DataFrame back to the same CSV file.
    df.to_csv(path_or_buf=csv_file_path, index=False)


def extract_gears_obs(dataset_name: str, datasets_dir_path: str) -> str:
    """Extract ['cell_id', 'condition_fixed'] from the GEARS datasets.

    Args:
        dataset_name: The name of the dataset.
        datasets_dir_path: The path to the datasets directory.

    Returns:
        The path to the CSV file containing the observations.
    """
    # Load the dataset.
    pert_dataset = pt.PertDataset(
        name=dataset_name, variant="gears", dir_path=datasets_dir_path
    )

    # Make the output file path.
    output_file_path = os.path.join(pert_dataset.path, "obs.csv")

    if not os.path.exists(output_file_path):
        # Export the observations.
        print(f"Exporting observations to: {output_file_path}")
        pert_dataset.export_obs_to_csv(
            file_path=output_file_path, obs=["cell_id", "condition_fixed"]
        )

        # Apply fixes.
        if dataset_name == "adamson":
            _fix_adamson(csv_file_path=output_file_path)
    else:
        print(f"Observations already exported to: {output_file_path}")

    return output_file_path
