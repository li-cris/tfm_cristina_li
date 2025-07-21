import argparse
import json
import logging
import os
import sys
import pickle
import random
import torch
import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, List


# SENA repo imports
# Found in ../SENA/src
from utils import Norman2019DataLoader # review Acuerdate de añadir utils a la carpeta de sena
from train import train

# Own sena-related imports
# Importable from pip install -e ../sypp
from sena.sena_utils import check_and_load_paths
from sena.inference import evaluate_double


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  # Logs to a file
        logging.StreamHandler(),  # Also logs to the console
    ],
)

# Opts has some default values that are overridden by the command line args
@dataclass
class Options:
    name: str = "norman_example" # args
    model: str = "sena"
    dataset_name: str = "Norman2019_reduced" # args
    batch_size: int = 32
    sena_lambda: float = 0 # args
    lr: float = 1e-3
    epochs: int = 100
    grad_clip: bool = False
    mxAlpha: float = 1.0
    mxBeta: float = 1.0
    mxTemp: float = 100.0
    lmbda: float = 0.1
    MMD_sigma: float = 200.0
    kernel_num: int = 10
    matched_IO: bool = False
    latdim: int = 105 # args
    seed: int = 42 # args
    num_runs: int = 1 # args
    dim: Optional[int] = None
    cdim: Optional[int] = None
    log: bool = False
    mlflow_port: int = 5678
    sena_eval_mode: List[str] = field(default_factory=lambda: ["double"]) # args
    numint: int = 2
    pool_size: int = 150
    top_deg: int = 100


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    # Directory where trained model and dataloaders are saved
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="./models/",
        help="Directory to save the trained models.",
    )

    # Directory where evaluation metric CSV files are saved for each named run
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./results/",
        help="Directory to save the evaluation metrics."
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the training."
    )

    parser.add_argument(
        "--model", type=str, default="sena", help="Model to use for training."
    )

    # Name given to the current project
    parser.add_argument("--name", type=str, default="example", help="Name of the run.")

    # Available datasets: Norman2019
    parser.add_argument("--dataset", type=str, default="Norman2019_reduced", help="Dataset for the run.")

    parser.add_argument("--latdim", type=int, default=105, help="Latent dimension.")

    # First seed used for the project
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--sena_lambda", type=float, default=0, help="Sena λ value")

    parser.add_argument(
        "--log", action='store_true', help="flow server log system"
    )

    # Number of iterations for each project. Following seeds are chosen by adding one to the previous seed.
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs for the experiment."
    )

    # Evaluation mode for SENA: double. It chooses the dataloader to evaluate.
    parser.add_argument(
        "--sena_eval_mode",
        nargs="+",
        default=["double"],
        help="Which folds of SENA to evaluate (train, test and/or double)"
    )

    # Number of interventions for SENA evaluation
    parser.add_argument(
        "--numint", type=int, default=2, help="Number of interventions for SENA evaluation.")

    parser.add_argument(
        "--pool_size", type=int, default=150, help="Control sample pool size for evaluation.")
    
    parser.add_argument(
        "--top_deg", type=int, default=100, help="Number of top differentially expressed genes to evaluate."
    )

    return parser.parse_args()


# Set seed for current run of the project
def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set up configuration for logging
def save_config(opts: Options, save_dir: str) -> None:
    """Save the configuration options to a JSON file."""
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(opts), f, indent=4)

# Save dataloader as pickle
def save_pickle(data: Any, filepath: str) -> None:
    """Save data to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args: argparse.Namespace) -> None:
    """Main master function."""
    device = args.device

    # Option overwrite
    opts = Options(
        epochs=args.epochs,
        latdim=args.latdim,
        seed=args.seed,
        num_runs=args.num_runs,
        model=args.model,
        sena_lambda=args.sena_lambda,
        name=args.name,
        dataset_name=args.dataset,
        log=args.log,
        sena_eval_mode=args.sena_eval_mode,
        numint=args.numint,
        pool_size=args.pool_size,
        top_deg=args.top_deg
    )

    logging.info(f"Configuration: {opts}")

    for current_run in range(opts.num_runs):

        # Set up current seed for specific run and name for current run
        current_seed = opts.seed + current_run
        run_name = f"{opts.name}_seed_{current_seed}_latdim_{opts.latdim}"
        set_seeds(current_seed)

        # Update directory for each run
        savedir = os.path.join(args.savedir, run_name)
        os.makedirs(savedir, exist_ok=True)


        logging.info(f"Loading {opts.dataset_name} dataset")
        # Dataset choice
        if 'Norman2019' in opts.dataset_name:
            data_handler = Norman2019DataLoader(batch_size=opts.batch_size, dataname=opts.dataset_name)

        else:
            raise ValueError(f"Dataset {opts.dataset_name} is not supported yet.")

        # Get data from single-gene perturbation
        (
            dataloader,
            dataloader2,
            dim,
            cdim,
            ptb_targets,
        ) = data_handler.get_data(mode="train")

        # Get data from double perturbation
        dataloader_double, _, _, _ = data_handler.get_data(mode="test")

        # Dimensions for SENA
        opts.dim = dim
        opts.cdim = cdim

        # I think this is not needed, but I leave it here for now
        if opts.latdim is None:
            opts.latdim = opts.cdim

        # Save configurations and data
        save_config(opts, savedir)
        save_pickle(ptb_targets, os.path.join(savedir, "ptb_targets.pkl"))
        save_pickle(dataloader2, os.path.join(savedir, "test_data_single_node.pkl"))
        save_pickle(dataloader, os.path.join(savedir, "train_data.pkl"))
        save_pickle(dataloader_double, os.path.join(savedir, "double_data.pkl"))

        # Review: currently only for double
        data_file_map = {
            "train": "train_data.pkl",
            "test": "test_data_single_node.pkl",
            "double": "double_data.pkl",
        }

        # Running process for SENA
        print(f"Initialising SENA run: {current_run+1}, seed: {current_seed}")
        train(dataloader=dataloader,
              opts=opts,
              device=device,
              savedir=savedir,
              logger=logging, # Review
              data_handler=data_handler)

        logging.info(f"Training completed for run {current_run + 1} with seed {current_seed}")
        # Predict and evaluate

        # Load the model and data for evaluation
        config_path = os.path.join(savedir, "config.json")
        model_path = os.path.join(savedir, "best_model.pt")
        # This is to load the appropriate pickle, 'double' by default
        data_path = os.path.join(savedir, data_file_map[opts.sena_eval_mode[0]])
        ptb_path = os.path.join(savedir, "ptb_targets.pkl")
        logging.info(f"Model and data saved in {savedir}.")

        # Loading required data from .pkl files for evaluation
        (dataloader, model, ptb_genes, config) = check_and_load_paths(model_path=model_path, 
                    data_path=data_path,
                    ptb_path=ptb_path,
                    config_path=config_path,
                    mode=opts.sena_eval_mode,
                    savedir=savedir)

        # Preparations for evaluation
        model = model.to(device)
        model.eval()

        # Given that it is still in SENA folder
        raw_data_path = os.path.join('../SENA/data',f"{opts.dataset_name}.h5ad")

        # Current model evaluation and saving metrics to the directory as CSV
        evaluate_double(
            model=model,
            dataloader=dataloader,
            data_path=raw_data_path,
            config=config,
            ptb_genes=ptb_genes,
            results_dir_path=args.eval_dir,
            run_name=run_name,
            device=device,
            numint=opts.numint
        )

        print("Process end.")


if __name__ == "__main__":
    args = parse_args()

    # Build the save directory path
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    args.eval_dir = os.path.join(args.eval_dir, args.name)
    os.makedirs(args.eval_dir, exist_ok=True)

    main(args)