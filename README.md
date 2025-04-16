## Linear Gene Expression Model

Train and test two versions of the linear gene expression model ("optimized" and "learned"):

```shell
python3 -m src.lgem.main
```

## SENA 2

### Training

SENA 2:

```shell
python3 -m src.sena2.main --model sena2
```

SENA-discrepancy-VAE:

```shell
python3 -m src.sena2.main --model sena
```

### Inference

```shell
python3 -m src.sena2.inference --savedir results/example --evaluation train test double
```

## Containers

We are using different Docker containers:

- `sena2_devcontainer`

    A [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) for developing with [Visual Studio Code](https://code.visualstudio.com).
    It is configured by the files in the folder [`.devcontainer`](.devcontainer).

- `sena2_mlcontainer`:

    A container providing a PyTorch environment with CUDA support for training, validating, and testing models.
    It is configured by the files in the folder [`docker`](docker).

    Build the Docker image:

    ```shell
    docker build --tag sena2_pytorch docker
    ```

    Run the Docker container:

    ```shell
    docker run --detach --tty --volume .:/workspace --gpus all --name sena2_mlcontainer sena2_pytorch
    ```

    Access the Docker container:

    ```shell
    docker exec --interactive --tty sena2_mlcontainer bash
    ```

    You can also try to install the container's pip environment in your native operating system:

    ```shell
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --requirement docker/requirements_other.txt
    pip install --requirement docker/requirements_torch.txt
    ```

## Data

We use a large-scale Perturb-seq dataset, which profiles gene expression changes in leukemia cells under genetic perturbations.
We use the preprocessed version by [CPA](https://github.com/facebookresearch/CPA):

```shell
wget https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar
mkdir cpa_binaries
tar -xvf cpa_binaries.tar -C cpa_binaries
cp cpa_binaries/datasets/Norman2019_raw.h5ad .
rm -r cpa_binaries
rm cpa_binaries.tar
```

Alternatively, we use the reduced version of this dataset from [SENA]:

```shell
wget https://raw.githubusercontent.com/ML4BM-Lab/SENA/refs/heads/master/data/Norman2019_reduced.zip
unzip Norman2019_reduced.zip
rm Norman2019_reduced.zip
```
