"""Utility to preprocess the raw data."""

import os
import sys

# Add the root of the project to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.norman import preprocess as norman_preprocess
from transmet.utils import get_git_root


def main() -> None:
    """Preprocess the raw data."""
    norman_preprocess(
        datasets_dir_path=os.path.join(get_git_root(), "datasets"),
        apply_gears_filter=True,
    )


if __name__ == "__main__":
    main()
