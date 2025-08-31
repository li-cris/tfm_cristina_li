import argparse
import numpy as np
import scanpy as sc
import os
import torch
from torch.utils.data import DataLoader

from sena.sena_utils import check_and_load_paths, find_pert_pairs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    # Directory where trained model and dataloaders are saved
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="./models/",
        help="Directory where models were saved.",
    )

    # Directory where evaluation metric CSV files are saved for each named run
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./results/",
        help="Directory to save the predictions."
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/",
        help="Directory from which dataset is loaded."
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the training."
    )

    # Name given to the current project
    parser.add_argument("--name", type=str, default="example", help="Name given to the project.")

    # Available datasets: Norman2019
    parser.add_argument("--dataset", type=str, default="Norman2019_reduced", help="Dataset for the run.")

    parser.add_argument("--latdim", type=int, default=105, help="Latent dimension of loaded model.")

    # First seed used for the project
    parser.add_argument("--seed", type=int, default=42, help="Seed used to train loaded model.")

    # Evaluation mode for SENA: double. It chooses the dataloader to evaluate.
    parser.add_argument(
        "--sena_eval_mode",
        nargs="+",
        default=["double"],
        help="Which folds of SENA to evaluate (train, test and/or double)"
    )

    parser.add_argument(
        "--pert_genes",
        nargs="+",
        default=["CEBPA", "CEBPB"],
        help="Gene perturbations to predict. Currently only for double perturbations."
    )

    # Number of interventions for SENA evaluation
    parser.add_argument(
        "--numint", type=int, default=2, help="Number of interventions for SENA evaluation.")

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for dataloader during prediction of GEPs. Each dataloader batch is 32 by default."
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Genes to predict
    genes = args.pert_genes

    # Directory where trained model and other data is saved.
    savedir = args.savedir

    # Specific trained model to get predictions from
    model_name = f"{args.name}_seed_{args.seed}_latdim_{args.latdim}"

    # Other parameters
    device = args.device
    batch_size = args.batch_size
    numint = args.numint
    sena_eval_mode = args.sena_eval_mode
    data_file_map = {
        "train": "train_data.pkl",
        "test": "test_data_single_node.pkl",
        "double": "double_data.pkl",
    }

    # Load raw adata
    # Get full path to load dataset
    if 'Norman2019' in args.dataset:
        dataset_path = os.path.join(args.dataset_dir, f"{args.dataset}.h5ad")

    # elif 'replogle' in args.dataset:
    #     dataset_path = os.path.join(args.dataset_dir, f"{args.dataset}", "perturb_processed.h5ad")

    else:
        raise ValueError(f"Dataset {args.dataset} is not supported yet.")

    # Load adata
    if os.path.exists(dataset_path):
        adata = sc.read_h5ad(dataset_path)
    else:
        raise FileNotFoundError(f"Data file not found at {dataset_path}")

    savedir = os.path.join(savedir, model_name)
    # Load the model and data for evaluation
    config_path = os.path.join(savedir, "config.json")
    model_path = os.path.join(savedir, "best_model.pt")
    # This is to load the appropriate pickle, 'double' by default
    data_path = os.path.join(savedir, data_file_map[sena_eval_mode[0]])
    ptb_path = os.path.join(savedir, "ptb_targets.pkl")

    # Loading required data from .pkl files for evaluation
    (dataloader, model, ptb_genes, config) = check_and_load_paths(model_path=model_path,
                data_path=data_path,
                ptb_path=ptb_path,
                config_path=config_path,
                mode=args.sena_eval_mode,
                savedir=savedir)
    temp = config.get("temp", 1000.0)

    # Get c_shape (intervention hot encoding matrix)
    all_pairs, _, c_shape = find_pert_pairs(dataloader=dataloader, device=device)

    # Load adata for given perturbation pair in raw adata. Only to get n.
    mask = (adata.obs['condition'] == "+".join([genes[0], genes[1]])) | (adata.obs['condition'] == "+".join([genes[1], genes[0]]))
    true_adata = adata[mask]
    n = true_adata.n_obs
    if n>200:
        n = 200

    # Load adata for control cells in raw adata. A portion is taken based on n.
    # And prepare control dataloader
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"].X.copy()
    ctrl_random = ctrl_adata[np.random.choice(ctrl_adata.shape[0], n, replace=False)]
    ctrl_geps_tensor = torch.tensor(ctrl_random.toarray()).double()
    ctrl_loader = DataLoader(ctrl_geps_tensor, batch_size=batch_size, shuffle=False)

    # Set c1 and c2 for prediction
    for num, unique_pairs in enumerate(all_pairs):
        gene_pair = "_".join([ptb_genes[unique_pairs[0]], ptb_genes[unique_pairs[1]]])
        if set(gene_pair.split("_")) == set(genes):
            c_shape_loader = c_shape[0, :].repeat(batch_size, 1)
            c1 = torch.zeros_like(c_shape_loader).double().to(device)
            c1[:, unique_pairs[0]] = 1
            c2 = torch.zeros_like(c_shape_loader).double().to(device)
            c2[:, unique_pairs[1]] = 1
            break

    # Prepare prediction data file
    var_names_str = ",".join(map(str, list(adata.var_names)))
    predictions_results_file_path = os.path.join(args.eval_dir, f"{gene_pair}_prediction_{model_name}.csv")
    print(f"File created as {gene_pair}_prediction_{model_name}.csv")

    # Predict for a given batch size of control cells and c1/c2 (interventions)
    # and save each batch of values to the prediction data file
    with open(file=predictions_results_file_path, mode="w") as f:
        print(f"{var_names_str}", file=f)
        for i, x in enumerate(ctrl_loader):
            x = x.to(device)

            if len(x) < len(c1):
                c1 = c1[:len(x), :]
                c2 = c2[:len(x), :]
            print(f"Predicting for batch {i+1}")
            with torch.no_grad():
                y_hat, _, _, _, _, _ = model(
                    x, c1, c2, num_interv=numint, temp=temp)

            y_hat_np = y_hat.detach().cpu().numpy()
            for gep in y_hat_np:
                print(",".join(map(str, gep)), file=f)
            print("Done.")


if __name__ == "__main__":
    args = parse_args()

    # Build the save directory path
    args.savedir = os.path.join(args.savedir, args.name)
    args.eval_dir = os.path.join(args.eval_dir, args.name)

    if os.path.exists(args.savedir) and os.path.exists(args.eval_dir):
        main(args)
    else:
        raise FileNotFoundError("Model or result not found at given directory and run name. Check that given arguments are correct.")