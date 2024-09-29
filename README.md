# TransMet

_Linking Perturbation Experiments to Metabolic Graphs Reveals Key Regulatory Mechanisms in Cellular Metabolism_

## Compass

A Docker setup to run [Compass](https://github.com/YosefLab/Compass), to characterize cellular metabolic states based on single-cell RNA sequencing and flux balance analysis, is provided in the separate repository [compass-docker](https://github.com/voges/compass-docker).

## Software Development

For your reference, we develop on Ubuntu 22.04.5 LTS with Python 3.10.12.

### Package and Environment Management

We use [pip](https://pip.pypa.io) for package and environment management.
Follow the steps below to set up your environment using the provided [`requirements.txt`](requirements.txt) file.

#### Setup Instructions

1. Create a virtual environment:
    ```sh
    python3 -m venv .venv
    ```

2. Activate the virtual environment:
    ```sh
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip3 install -r requirements.txt
    ```

#### Additional Commands

- Install additional packages:
    ```sh
    pip3 install <package>
    ```

- Update the requirements file:
    ```sh
    pip3 freeze > requirements.txt
    ```

- Deactivate the virtual environment:
    ```sh
    deactivate
    ```

### Git Hooks

We use [pre-commit](https://pre-commit.com) to automatically run checks on every commit.

1. Install the pre-commit package manager:
    ```sh
    pip3 install pre-commit
    ```
2. Set up the Git hook scripts specified in [`.pre-commit-config.yaml`](.pre-commit-config.yaml):
    ```sh
    pre-commit install
    ```
3. Run against all files (optional):
    ```sh
    pre-commit run --all files
    ```

### Code Linting

We use [Ruff](https://github.com/astral-sh/ruff) to check the code for linting issues.

1. Install Ruff:
    ```sh
    pip3 install ruff
    ```

2. Run the following command from the root of the Git repository:
    ```sh
    ruff check .
    ```
