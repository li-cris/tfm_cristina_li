# TransMet

_Linking Perturbation Experiments to Metabolic Graphs Reveals Key Regulatory Mechanisms in Cellular Metabolism_

## Git Hooks

We use [pre-commit](https://pre-commit.com) to automatically run checks on every commmit.

- Install the pre-commit package manager:
    ```sh
    pip3 install pre-commit
    ```
- Set up the git hook scripts specified in `.pre-commit-config.yaml`:
    ```sh
    pre-commit install
    ```
- Run against all files (optional):
    ```sh
    pre-commit run --all files
    ```

## Package and Environment Management

We use [pip](https://pip.pypa.io) for package and environment management.
Follow the steps below to set up your environment using the provided [`requirements.txt`](requirements.txt) file.

### Setup Instructions

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

### Additional Commands

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

## Code Linting

To check the code for linting issues, run the following command from the root of the git repository:

```sh
ruff check .
```
