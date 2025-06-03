import argparse
import itertools
import os
import sys

path = os.path.abspath('../sypp/src/gears_tools/')
print(path)
sys.path.insert(0, path)

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mmd_loss import MMDLoss
from kld_loss import compute_kld
from scipy.sparse import csr_matrix

from gears import GEARS, PertData

DATA_DIR_PATH = "data"
MODELS_DIR_PATH = "models"
RESULTS_DIR_PATH = "results"

PREDICT_SINGLE = False
PREDICT_DOUBLE = True
# PREDICT_COMBO = False

# Set to True if training only has single perturbations and double perturbations are on test.
SINGLE_TRAIN_ONLY = True


def train(
    pert_data: PertData, dataset_name: str, split: str, seed: int, hidden_size: int, device: str
) -> str:
    """Set up, train, and save GEARS model."""
    print("Training GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.model_initialize(hidden_size=hidden_size)
    gears_model.train(epochs=20)
    model_name = (
        f"gears_{dataset_name}_split_{split}_seed_{str(seed)}_hidden_size_{str(hidden_size)}"
    )
    gears_model.save_model(path=os.path.join(MODELS_DIR_PATH, model_name))
    return model_name


def predict(pert_data: PertData, device: str, model_name: str) -> None:
    """Predict with GEARS model."""
    # Load the model.
    print("Loading GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.load_pretrained(path=os.path.join(MODELS_DIR_PATH, model_name))

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
            RESULTS_DIR_PATH, f"{model_name}_single.csv"
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
            RESULTS_DIR_PATH, f"{model_name}_double.csv"
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

    # if PREDICT_COMBO:
    #     # Predict all combo perturbations.
    #     combo_results_file_path = os.path.join(
    #         RESULTS_DIR_PATH, f"{model_name}_combo.csv"
    #     )
    #     with open(file=combo_results_file_path, mode="w") as f:
    #         print(f"combo,{var_names_str}", file=f)
    #         for i, c in enumerate(combo_perturbations):
    #             print(f"Predicting combo {i + 1}/{len(combo_perturbations)}: {c}")
    #             prediction = gears_model.predict(pert_list=[c])
    #             combo = next(iter(prediction.keys()))
    #             expressions = prediction[combo]
    #             expressions_str = ",".join(map(str, expressions))
    #             print(f"{combo},{expressions_str}", file=f)

def evaluate_double(adata: AnnData, model_name: str, pool_size: int = 250) -> None:
    """Evaluate the predicted GEPs of double perturbations."""
    # Load predicted GEPs.
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(RESULTS_DIR_PATH, f"{model_name}_double.csv")
    )

    # Make results file path.
    results_file_path = os.path.join(
        RESULTS_DIR_PATH, f"{model_name}_double_change_metrics.csv"
    )

    with open(file=results_file_path, mode="w") as f:
        print(
            "double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred",
            file=f,
        )

        for i, double in enumerate(df["double"]):
            # Get the predicted GEP for the current double perturbation.
            pred_geps = df.loc[df["double"] == double]
            pred_geps = pred_geps.iloc[0, 1:].tolist()
            pred_geps = np.array([pred_geps])

            # Get all the true GEPs with the current double perturbation.
            double = double.replace("_", "+")
            print(f"Evaluating double {i + 1}/{len(df['double'])}: {double}")
            true_geps = adata[adata.obs["condition"] == double]

            # Limiting n
            if true_geps.n_obs>pool_size:
                n = pool_size
                random_indices = np.random.choice(true_geps.n_obs, size=n, replace=False)
                true_geps = true_geps[random_indices, :]   
            else:
                # If less than pool size, randomly sample from all true_geps to avoid error in MMD computation
                n = true_geps.n_obs

            # Obtaining random sample of ctrl GEP
            all_ctrl_geps = adata[adata.obs["condition"] == "ctrl"]
            random_indices = np.random.choice(
                all_ctrl_geps.n_obs, size=n, replace=False
            )
            ctrl_geps = all_ctrl_geps[random_indices, :]
            pred_geps = csr_matrix(np.tile(pred_geps, reps=(n, 1)))

            # Another random ctrl_gep
            random_indices_2 = np.random.choice(
                all_ctrl_geps.n_obs, size=n, replace=False
            )
            ctrl_geps_2 = all_ctrl_geps[random_indices_2, :]

            # Tensor conversion and differential expression
            ctrl_geps_tensor = torch.tensor(ctrl_geps.X.toarray())
            ctrl_ctrl_geps_tensor = torch.tensor(ctrl_geps_2.X.toarray()) - ctrl_geps_tensor
            true_ctrl_geps_tensor = torch.tensor(true_geps.X.toarray()) - ctrl_geps_tensor
            pred_ctrl_geps_tensor = torch.tensor(pred_geps.toarray()) - ctrl_geps_tensor

            # MMD setup.
            mmd_sigma = 200.0
            kernel_num = 10
            mmd_loss = MMDLoss(fix_sigma=mmd_sigma, kernel_num=kernel_num)

            # Compute MMD 
            mmd_true_vs_ctrl = mmd_loss.forward(
                            source=ctrl_ctrl_geps_tensor, target=true_ctrl_geps_tensor
                        )

            mmd_true_vs_pred = mmd_loss.forward(
                source=pred_ctrl_geps_tensor, target=true_ctrl_geps_tensor
            )

            # Compute MSE
            mse_true_vs_ctrl = torch.mean(
                (true_ctrl_geps_tensor - ctrl_ctrl_geps_tensor) ** 2
            ).item()
            mse_true_vs_pred = torch.mean(
                (true_ctrl_geps_tensor - pred_ctrl_geps_tensor) ** 2
            ).item()

            # Compute KLD
            kld_true_vs_ctrl = compute_kld(true_ctrl_geps_tensor, ctrl_ctrl_geps_tensor)
            kld_true_vs_pred = compute_kld(true_ctrl_geps_tensor, pred_ctrl_geps_tensor)

            print(f"MMD (true vs. control):   {mmd_true_vs_ctrl:10.6f}")
            print(f"MMD (true vs. predicted): {mmd_true_vs_pred:10.6f}")
            print(f"MSE (true vs. control):   {mse_true_vs_ctrl:10.6f}")
            print(f"MSE (true vs. predicted): {mse_true_vs_pred:10.6f}")
            print(f"KLD (true vs. control):   {kld_true_vs_ctrl:10.6f}")
            print(f"KLD (true vs. predicted): {kld_true_vs_pred:10.6f}")
            print(
                f"{double},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred},{kld_true_vs_ctrl},{kld_true_vs_pred}",
                file=f,
            )


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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use."
    )
    parser.add_argument("--model_name", type=str, default="gearsmodel", help="Custom name for the model."
    )
    parser.add_argument("--dataset_name", type=str, default="norman", help="Name of dataset in DATA_DIR_PATH."
    )
    parser.add_argument("--pool_size", type=int, default=250, help="Size of the pool for evaluation."
    )
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to perform with different seeds."
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
        # This split of train test sizes keeps singles in training and doubles in testing
        print("Preparing data split.")
        if SINGLE_TRAIN_ONLY:
            # If training only has single perturbations, we need to set the train_gene_set_size
            # to 1.0 and combo_seen2_train_frac to 0.0.
            print("Keeping only single perturbation samples in training.")
            pertdata.prepare_split(
                split=args.split,
                seed=current_seed,
                train_gene_set_size=1.0,
                combo_seen2_train_frac=0.0,
            )
        else:
            pertdata.prepare_split(split=args.split, seed=current_seed)
        pertdata.get_dataloader(batch_size=32)

        # Train.
        model_name = train(
            pert_data=pertdata,
            dataset_name=args.model_name,
            split=args.split,
            seed=current_seed,
            hidden_size=args.hidden_size,
            device=args.device,
        )

        # Predict.
        predict(pert_data=pertdata, device=args.device, model_name=model_name)

        # Evaluate.
        pertdata = PertData(data_path=DATA_DIR_PATH)
        pertdata.load(data_name=data_name)
        evaluate_double(adata=pertdata.adata, model_name=model_name, pool_size=args.pool_size)


if __name__ == "__main__":
    main()