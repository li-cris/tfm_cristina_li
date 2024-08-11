"""Utility functions."""

import logging
import os
import subprocess
from typing import Optional

from omegaconf import DictConfig, OmegaConf
import requests
from tqdm import tqdm
from zipfile import ZipFile

log = logging.getLogger(__name__)


def log_yaml(yaml_dump: str) -> None:
    """
    Pretty log of a YAML dump.

    Args:
        yaml_dump: The YAML dump to log.
    """
    for line in yaml_dump.splitlines():
        log.info(f"{line}")


def log_config(cfg: DictConfig) -> None:
    """
    Pretty log of a Hydra configuration.

    Args:
        cfg: The Hydra configuration.
    """
    log.info("")
    log.info("Configuration:")
    log.info("--------------")
    log_yaml(yaml_dump=OmegaConf.to_yaml(cfg=cfg))
    log.info("")


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
        log.error(f"Failed to get Git root: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        return None


def download_file(url: str, save_filename: str) -> None:
    """
    Download of a file with progress bar.

    The progress bar will display the size in binary units (e.g., KiB for kibibytes,
    MiB for mebibytes, GiB for gibibytes, etc.), which are based on powers of 1024.

    Args:
        url: The URL of the data.
        save_filename: The path to save the data.
    """
    if not os.path.exists(path=save_filename):
        log.info(f"Downloading: {url}")
        try:
            with requests.get(url=url, stream=True) as response:
                response.raise_for_status()
                total_size_in_bytes = int(
                    response.headers.get(key="content-length", default=0)
                )
                log.info(f"Total size: {total_size_in_bytes:,} bytes")
                block_size = 1024
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(file=save_filename, mode="wb") as file:
                    for data in response.iter_content(chunk_size=block_size):
                        progress_bar.update(n=len(data))
                        file.write(data)
                progress_bar.close()
        except requests.exceptions.RequestException as e:
            log.error(f"Error downloading file: {e}")
            if "progress_bar" in locals():
                progress_bar.close()
    else:
        log.info(f"Found local copy: {save_filename}")


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract a ZIP file.

    Args:
        zip_path: The path to the ZIP file.
        extract_dir: The directory to extract the ZIP file into.
    """
    if not os.path.exists(path=extract_dir):
        log.info(f"Extracting ZIP file: {zip_path}")
        with ZipFile(file=zip_path) as zipfile:
            zipfile.extractall(path=extract_dir)
    else:
        log.info(f"Found extracted files: {extract_dir}")
