import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import random

# GEARS imports
# from gears import GEARS, PertData (Slight change in PertData)
from gears import GEARS

# Own imports
from gears_tools.pertdata import PertData
from gears_tools.inference import evaluate_double, evaluate_single


DATA_DIR_PATH = "data"
MODELS_DIR_PATH = "models"
RESULTS_DIR_PATH = "results"

PREDICT_SINGLE = True
PREDICT_DOUBLE = False

# Set to True if training only has single perturbations (train + val) and double perturbations are on test.
SINGLE_TRAIN_ONLY = True

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(
    pert_data: PertData, dataset_name: str, model_savedir: str,
    split: str, seed: int, hidden_size: int, decoder_hidden_size: int,
    device: str, epochs: int = 20
) -> str:
    """Set up, train, and save GEARS model."""
    print("Training GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.model_initialize(hidden_size=hidden_size, decoder_hidden_size=decoder_hidden_size)
    gears_model.train(epochs=epochs)
    model_name = (
        f"gears_{dataset_name}_split_{split}_seed_{str(seed)}_hidden_size_{str(hidden_size)}_decoder_hidden_size_{str(decoder_hidden_size)}"
    )
    gears_model.save_model(path=os.path.join(model_savedir, model_name))
    return model_name


def predict(pert_data: PertData,
            device: str, model_name: str,
            model_savedir: str, results_savedir: str) -> None:
    """Predict with GEARS model."""
    # Load the model.
    print("Loading GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.load_pretrained(path=os.path.join(model_savedir, model_name))

    # Get all single perturbations.
    single_perturbations = set(
        [
            c.strip("+ctrl")
            for c in pert_data.adata.obs["condition"]
            if ("ctrl+" in c) or ("+ctrl" in c)
        ]
    )
    print(f"Number of single perturbations: {len(single_perturbations)}")

    # Get all double perturbations.
    double_perturbations = set(
        [c for c in pert_data.adata.obs["condition"] if "ctrl" not in c]
    )
    print(f"Number of double perturbations: {len(double_perturbations)}")

    # Generate all possible double perturbations (combos).
    combo_perturbations = []
    for g1 in single_perturbations:
        for g2 in single_perturbations:
            if g1 == g2:
                continue
            combo_perturbations.append(sorted([g1, g2]))
    combo_perturbations.sort()
    combo_perturbations = list(k for k, _ in itertools.groupby(combo_perturbations))
    print(f"Number of combo perturbations: {len(combo_perturbations)}")

    # Get the names of all measured genes as comma-separated list.
    var_names_str = ",".join(map(str, list(pert_data.adata.var_names)))

    if PREDICT_SINGLE:
        # Predict all single perturbations.
        single_results_file_path = os.path.join(
            results_savedir, f"{model_name}_single.csv"
        )
        with open(file=single_results_file_path, mode="w") as f:
            print(f"single,{var_names_str}", file=f)
            for i, g in enumerate(single_perturbations):
                print(f"Predicting single {i + 1}/{len(single_perturbations)}: {g}")
                prediction = gears_model.predict(pert_list=[[g]])
                single = next(iter(prediction.keys()))
                expressions = prediction[single]
                expressions_str = ",".join(map(str, expressions))
                print(f"{single},{expressions_str}", file=f)

    if PREDICT_DOUBLE:
        # Predict all double perturbations.
        double_results_file_path = os.path.join(
            results_savedir, f"{model_name}_double.csv"
        )
        with open(file=double_results_file_path, mode="w") as f:
            print(f"double,{var_names_str}", file=f)
            for i, d in enumerate(double_perturbations):
                print(f"Predicting double {i + 1}/{len(double_perturbations)}: {d}")
                prediction = gears_model.predict(pert_list=[d.split("+")])
                double = next(iter(prediction.keys()))
                expressions = prediction[double]
                expressions_str = ",".join(map(str, expressions))
                print(f"{double},{expressions_str}", file=f)


def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Train, predict, evaluate GEARS.")
    parser.add_argument("--split", type=str, default="no_test", help="Data split.")
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for data splitting."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Size of the hidden layers."
    )
    parser.add_argument(
        "--decoder_hidden_size", type=int, default=16, help="Size of the hidden decoder layers."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use."
    )
    parser.add_argument("--project_name", type=str, default="gears", help="Custom name for the project."
    )
    parser.add_argument("--dataset_name", type=str, default="norman", help="Name of dataset in DATA_DIR_PATH."
    )
    parser.add_argument("--pool_size", type=int, default=250, help="Size of the pool for evaluation."
    )
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform with different seeds."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs. Default is 20."
    )
    parser.add_argument("--top_deg", type=int, default=20, help="Number of Top Differentially Expressed Genes to evalaute."
    )
    args = parser.parse_args()

    # Print all arguments.
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create directories.
    os.makedirs(name=DATA_DIR_PATH, exist_ok=True)
    os.makedirs(name=MODELS_DIR_PATH, exist_ok=True)
    os.makedirs(name=RESULTS_DIR_PATH, exist_ok=True)

    # Create directory for custom name in MODELS_DIR_PATH and RESULTS_DIR_PATH.
    model_savedir = os.path.join(MODELS_DIR_PATH, args.project_name)
    results_savedir = os.path.join(RESULTS_DIR_PATH, args.project_name)
    os.makedirs(name=model_savedir, exist_ok=True)
    os.makedirs(name=results_savedir, exist_ok=True)

    # Load "norman" data.
    data_name = args.dataset_name
    print(f"Loading '{data_name}' data.")
    pertdata = PertData(DATA_DIR_PATH)
    pertdata.load(data_path=os.path.join(DATA_DIR_PATH, data_name))

    seed = args.seed

    # Running train, predict, evaluate for multiple runs.
    for current_run in range(args.num_runs):
        # Update the seed for each run.
        current_seed = seed + current_run
        print(f"Current run: {current_run + 1}/{args.num_runs}, Seed: {current_seed}")

        # Split data and get dataloaders. This is the same procedure as used for Figure 4 in
        # the GEARS paper.
        # See also: https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py
        # This split of train test sizes keeps singles in training and validation and doubles in testing
        print("Preparing data split.")

        # Condition to evaluate only single perturbation dataset
        if args.split == 'simulation_single':
            print("Training and evaluating with single perturbation data.")
            pertdata.adata = pertdata.adata[pertdata.adata.obs['condition'].str.contains('ctrl')]
            pertdata.prepare_split(split=args.split, seed=current_seed)

        # Condition to evaluate double perturbation dataset with single perturbations in training
        elif SINGLE_TRAIN_ONLY and args.split != 'simulation_single':
            # If training only has single perturbations, we need to set the train_gene_set_size
            # to 1.0 and combo_seen2_train_frac to 0.0.
            print("Keeping only single perturbation samples in training.")
            pertdata.prepare_split(
                split=args.split,
                seed=current_seed,
                train_gene_set_size=1.0,
                combo_seen2_train_frac=0.0
            )

        # Condition to evaluate double perturbation dataset with both types in training
        else:
            print("Training with both single and double perturbation data and evaluating with double perturbations.")
            pertdata.prepare_split(split=args.split, seed=current_seed)
        pertdata.get_dataloader(batch_size=32)

        # Train.
        model_name = train(
            pert_data=pertdata,
            dataset_name=args.dataset_name,
            model_savedir=model_savedir,
            split=args.split,
            seed=current_seed,
            hidden_size=args.hidden_size,
            decoder_hidden_size=args.decoder_hidden_size,
            device=args.device,
            epochs=args.epochs
        )

        # Predict.
        predict(pert_data=pertdata, device=args.device, model_name=model_name,
                model_savedir=model_savedir, results_savedir=results_savedir)

        # Evaluate.
        pertdata = PertData(DATA_DIR_PATH)
        pertdata.load(data_path=os.path.join(DATA_DIR_PATH, data_name))
        if PREDICT_DOUBLE:
            evaluate_double(adata=pertdata.adata, model_name=model_name,
                            results_savedir=results_savedir, pool_size=args.pool_size,
                            seed=current_seed, top_deg=args.top_deg)
        if PREDICT_SINGLE:
            evaluate_single(adata=pertdata.adata, model_name=model_name,
                            results_savedir=results_savedir, pool_size=args.pool_size,
                            seed=current_seed, top_deg=args.top_deg)

if __name__ == "__main__":
    main()