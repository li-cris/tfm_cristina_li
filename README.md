# Transformer-Based Activity Discrepancy Variational Autoencoder

## Docker

We are using different Docker containers:

- `tadvae_devcontainer`

    A [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) for developing with [Visual Studio Code](https://code.visualstudio.com).
    It is configured by the files in the folder [`.devcontainer`](.devcontainer).

- `tadvae_mlcontainer`:

    A container providing a PyTorch environment with CUDA support for training, validating, and testing models.
    It is configured by the files in the folder [`docker`](docker).

    Build the Docker image:

    ```shell
    docker build --tag tadvae_pytorch docker
    ```

    Run the Docker container:

    ```shell
    docker run --detach --tty --volume $(git rev-parse --show-toplevel):/workspace --gpus all --name tadvae_mlcontainer --user $(id --user):$(id --group) tadvae_pytorch
    ```

    Access the Docker container:

    ```shell
    docker exec --interactive --tty tadvae_mlcontainer bash
    ```
