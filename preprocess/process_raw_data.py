"""Functionality to process the raw data into the H5AD format."""

import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.norman import preprocess as norman_preprocess
from transmet.utils import get_git_root


def main() -> None:
    norman_preprocess(datasets_path=os.path.join(get_git_root(), "datasets"))


if __name__ == "__main__":
    main()
