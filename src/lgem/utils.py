import git


def get_git_root() -> str:
    """Get the root directory of the Git repository."""
    return git.Repo(search_parent_directories=True).working_tree_dir
