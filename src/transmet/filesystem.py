"""Filesystem utilities."""

import subprocess


def get_git_root() -> str:
    """Return the root directory of the current Git repository.

    Returns:
        The root directory of the current Git repository, or "" if the command fails.
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
    return ""
