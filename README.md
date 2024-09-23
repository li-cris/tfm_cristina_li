# TransMet

## Getting Started

Begin by exploring the [`data_tutorial.ipynb`](data_tutorial.ipynb) to familiarize yourself with the perturbation data.

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
