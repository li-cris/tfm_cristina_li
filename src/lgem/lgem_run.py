import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import json

from typing import List

import os
# import sys
# path = os.path.abspath('../sypp/src/')
# print(path)
# sys.path.insert(0, path)
# sys.path.insert(0, os.path.abspath('../sypp/src/lgem/'))
# sys.path.insert(0, os.path.abspath('../'))

from lgem.models import (
    LinearGeneExpressionModelLearned,
    LinearGeneExpressionModelOptimized,
)
from lgem.test import test as test_lgem
from lgem.train import train as train_lgem
from lgem.data import compute_embeddings_double, pseudobulk_data_per_perturbation
from data_utils.single_norman_utils import separate_data, get_common_genes   



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    # Directory where trained model and dataloaders are saved
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="./models/",
        help="Main directory to save the trained models."
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./results/",
        help="Main directory to save prediction results and evaluation metrics."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Directory where the dataset is stored. Best to use global path here."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="lgem",
        help="Name to give the project folder in main directories."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Norman_2019raw",
        help="Name of the dataset to use for training and evaluation. Needs to be .h5ad file."
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Chosen device in which run model."
    ) 

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )

    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs."
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of epochs."
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size."
    )

    parser.add_argument(
        "--prediction_type", type=str, default="double", help="Kind of prediction to carry out in evaluation. Supported: 'double' or other."
    )

    return parser.parse_args()


def save_config(args: argparse.Namespace, filepath: str) -> None:
    """
    Save the configuration to a file.
    Parameters:
        args (argparse.Namespace): Parsed command line arguments.
        filepath (str): Path to save the configuration file.
    """
    config = vars(args)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filepath}")


def main(args):

    criterion = nn.MSELoss()
    seed=args.seed
    num_runs=args.num_runs
    n_epochs=args.epochs
    batch_size=args.batch_size
    run_name = f"lgem_{args.dataset_name}_epochs_{n_epochs}_batch_{batch_size}"
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(seed)

    # Print all arguments.
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Save args to a file
    args_file_path = os.path.join(args.savedir, "config.json")
    save_config(args, args_file_path)

    # Make results file path.
    results_file_path = os.path.join(
        args.eval_dir, f"{run_name}_train_test_metrics.csv"
        )

    global_data_path=os.path.join(args.data_dir, args.dataset_name + ".h5ad")
    print(f"Loading dataset from {global_data_path}.")
    pertdata = sc.read(global_data_path)
    prediction_type = args.prediction_type
    dataset_name = args.dataset_name

    # Get separate dataset for single, double perts and controls
    pertdata_single, pertdata_double, pertdata_ctrl = separate_data(adata = pertdata, dataset_name = dataset_name)

    if prediction_type == "double":
        # Join both AnnData datasets
        pertdata_both = pertdata_single.concatenate(pertdata_double, join = 'outer', index_unique = '-')
        all_perts, perts, genes, pertdata_common = get_common_genes(adata = pertdata_both, dataset_name = dataset_name)
    else:
        # Y, perts and genes
        all_perts, perts, genes, pertdata_common = get_common_genes(adata_single = pertdata_single, dataset_name = dataset_name)    

    print(f"Number of unique perturbations: {len(perts)}/{len(all_perts)}")

    Y = pseudobulk_data_per_perturbation(perts, genes, pertdata_common)

    # Ordered perturbations equivalent to Y is in perts
    # Compute the embeddings on the entire dataset (with singles and doubles)
    # G, P, b = compute_embeddings(Y.T, perts, genes)  # noqa: N806
    G, P, b = compute_embeddings_double(Y.T, perts, genes)  # noqa: N806

    # Keeping only embedding related to single perturbations for training
    singles_idx = [i for i, pert in enumerate(perts) if "+" not in pert]
    sY = Y[singles_idx, :]
    sP = P[singles_idx, :]

    with open(file=results_file_path, mode="w") as f:
        print(
            "seed,optimized_train_loss,optimized_model_loss,learned_train_loss,learned_model_loss",
            file=f,
        )

        for current_run in range(num_runs):
            current_seed = seed + current_run # Redundant in double prediction
            torch.manual_seed(current_seed)
            model_name = f"lgem_{dataset_name}_seed_{current_seed}_epoch_{n_epochs}_batch_{batch_size}"

            # Directory for custom name for model where more pickle files will be saved
            # savedir = args.savedir
            savedir = os.path.join(args.savedir, model_name)
            os.makedirs(savedir, exist_ok=True)

            if prediction_type != "double":
                # Split the data into training and test sets and create the dataloaders.
                Y_train, Y_test, P_train, P_test = train_test_split(  # noqa: N806
                    Y, P, test_size=0.2, random_state=current_seed
                )
                # Get indices of perturbations in the train/test sets
                train_indices, test_indices = train_test_split(range(len(perts)), test_size=0.2, random_state=42)
                # Get the equivalent perturbations for the training and test sets
                perts_train = [perts[i] for i in train_indices]
                perts_test = [perts[i] for i in test_indices]

                train_dataset = TensorDataset(P_train, Y_train)
                test_dataset = TensorDataset(P_test, Y_test)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # Saving dataloaders as pickles and perturbation order from perts
                torch.save(train_dataloader, os.path.join(savedir, "train_dataloader.pt"))
                torch.save(test_dataloader, os.path.join(savedir, "train_dataloader.pt"))
                torch.save({"perts_train": perts_train,
                            "perts_test": perts_test},
                            os.path.join(savedir, "perts.pt"))
                val_dataloader = None

            else:
                # Keep train and val dataloaders for training
                Y_train, Y_val, P_train, P_val = train_test_split(  # noqa: N806
                    sY, sP, test_size=0.2, random_state=current_seed
                )

                # Train
                train_dataset = TensorDataset(P_train, Y_train)
                val_dataset = TensorDataset(P_val, Y_val)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


                # Test# Get indices of perturbations in the train/test set
                doubles_idx = [i for i, pert in enumerate(perts) if "+" in pert]
                doubles_dataset = TensorDataset(P[doubles_idx, :], Y[doubles_idx, :])
                test_dataloader = DataLoader(doubles_dataset, batch_size=batch_size, shuffle=False)

                # Save dataloaders as pickles and perturbation order from perts
                torch.save(train_dataloader, os.path.join(savedir, "train_dataloader.pt"))
                torch.save(val_dataloader, os.path.join(savedir, "val_dataloader.pt"))
                torch.save(test_dataloader, os.path.join(savedir, "test_dataloader.pt")) # equivalent to doubles only
                torch.save({"perts": perts}, os.path.join(savedir, "perts.pt"))

            # Fit the optimized model to the training data.
            model_optimized = LinearGeneExpressionModelOptimized(Y_train.T, G, P_train, b)
            train_loss_op = test_lgem(model_optimized, criterion, train_dataloader, device)
            print(f"Train loss (optimized model) | Run {current_run+1}: {train_loss_op:.4f}")
            test_loss_op = 0

            if prediction_type != "double":
                # Test the optimized model.
                test_loss_op = test_lgem(model_optimized, criterion, test_dataloader, device)
                print(f"Val loss (optimized model): {test_loss_op:.4f}")
            else:
                # Test the optimized model on validation set
                test_loss_op = test_lgem(model_optimized, criterion, val_dataloader, device)
                print(f"Val loss (optimized model): {test_loss_op:.4f}")

            # Fit the learned model to the training data.
            model_learned = LinearGeneExpressionModelLearned(G, b)
            optimizer = torch.optim.Adam(params=model_learned.parameters(), lr=1e-3)
            model_learned = train_lgem(
                model_learned, criterion, optimizer, train_dataloader, val_dataloader, n_epochs, device, validation=True
            )
            train_loss_learn = test_lgem(model_optimized, criterion, val_dataloader, device)
            print(f"Train loss (optimized model) | Run {current_run+1}: {train_loss_op:.4f}")
            test_loss_learn = 0

            if prediction_type != "double":
                # Test the learned model.
                test_loss_learn = test_lgem(model_learned, criterion, test_dataloader, device)
                print("Val loss (learned model): {test_loss_learn:.4f}")  
            else:
                test_loss_learn = test_lgem(model_learned, criterion, val_dataloader, device)
                print(f"Val loss (learned model): {test_loss_learn:.4f}")

            # Save results to file
            print(f"{current_seed},{train_loss_op}, {test_loss_op},{train_loss_learn}, {test_loss_learn}",
                file=f,
            )

            # Save models and embeddings
            torch.save(model_optimized.state_dict(), os.path.join(savedir, "optimized_best_model.pt"))
            torch.save(model_learned.state_dict(), os.path.join(savedir, "learned_best_model.pt"))
            torch.save(G, os.path.join(savedir, "G.pt"))
            torch.save(b, os.path.join(savedir, "b.pt"))
            torch.save(P_train, os.path.join(savedir, "P.pt"))
            torch.save(Y_train, os.path.join(savedir, "Y.pt"))
            print(f"Saved models and embedding at {savedir}")



if __name__ == "__main__":
    args = parse_args()

    # Build the save directory path
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    args.eval_dir = os.path.join(args.eval_dir, args.name)
    os.makedirs(args.eval_dir, exist_ok=True)

    main(args)