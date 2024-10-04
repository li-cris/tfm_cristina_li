"""Utility functions."""

import os
import subprocess
import tarfile
from typing import Optional
from zipfile import ZipFile

import requests
from tqdm import tqdm


def get_git_root() -> Optional[str]:
    """
    Return the root directory of the current Git repository.

    Returns:
        The root directory of the current Git repository, or None if the command fails.
    """
    try:
        return subprocess.check_output(
            args=["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get Git root: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def download_file(url: str, path: str) -> None:
    """
    Download a file with a progress bar.

    The progress bar will display the size in binary units (e.g., KiB for kibibytes,
    MiB for mebibytes, GiB for gibibytes, etc.), which are based on powers of 1024.

    Args:
        url: The URL of the data.
        path: The path to save the data.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP
            request.
        OSError: If there is an issue with writing the file.
    """
    if not os.path.exists(path=path):
        print(f"Downloading: {url} -> {path}")
        try:
            with requests.get(url=url, stream=True) as response:
                response.raise_for_status()
                total_size_in_bytes = int(
                    response.headers.get(key="content-length", default=0)
                )
                print(f"Total size: {total_size_in_bytes:,} bytes")
                block_size = 1024
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(file=path, mode="wb") as file:
                    for data in response.iter_content(chunk_size=block_size):
                        progress_bar.update(n=len(data))
                        file.write(data)
                progress_bar.close()
            print(f"Download completed: {path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
        except OSError as e:
            print(f"Error writing file: {e}")
        finally:
            if "progress_bar" in locals():
                progress_bar.close()
    else:
        print(f"File already exists: {path}")


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract a ZIP file.

    Args:
        zip_path: The path to the ZIP file.
        extract_dir: The directory to extract the ZIP file into.

    Raises:
        ValueError: If there is an issue extracting the ZIP file.
    """
    print(f"Extracting ZIP file: {zip_path} -> {extract_dir}")
    try:
        with ZipFile(file=zip_path) as f:
            f.extractall(path=extract_dir)
        print(f"Extraction completed: {extract_dir}")
    except f.BadZipFile as e:
        raise ValueError(f"Error extracting ZIP file: {e}")


def extract_tar(tar_path: str, extract_dir: str) -> None:
    """
    Extract a TAR file.

    Args:
        tar_path: The path to the TAR file.
        extract_dir: The directory to extract the TAR file into.

    Raises:
        ValueError: If there is an issue extracting the TAR file.
    """
    print(f"Extracting TAR file: {tar_path} -> {extract_dir}")
    try:
        with tarfile.open(name=tar_path) as f:
            f.extractall(path=extract_dir)
        print(f"Extraction completed: {extract_dir}")
    except f.TarError as e:
        raise ValueError(f"Error extracting TAR file: {e}")


def ensembl_id_to_gene_name(ensembl_id: str) -> str:
    """
    Get the gene name for an Ensembl ID using the Ensembl REST API.

    To map Ensembl stable IDs (such as gene, transcript, or protein IDs) to gene names
    (i.e., HGNC symbols from the [HUGO Gene Nomenclature Committee](https://www.genenames.org)),
    we use Ensembl's REST API.

    Args:
        ensembl_id: The Ensembl ID.

    Returns:
        The gene name.

    Raises:
        requests.exceptions.Request: If the request to the Ensembl REST API fails.
    """
    # The URL for the Ensembl REST API
    url = (
        f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
    )

    try:
        # Send a GET request to the Ensembl REST API
        response = requests.get(url=url)
        response.raise_for_status()

        # Return the gene name from the JSON response
        data = response.json()
        return data.get("display_name", "No gene name found")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to get gene name from Ensembl REST API: {e}")
