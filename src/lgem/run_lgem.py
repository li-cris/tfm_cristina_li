import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
from tqdm import tqdm

import os
import sys
path = os.path.abspath('../sypp/src/')
print(path)
sys.path.insert(0, path)
sys.path.insert(0, os.path.abspath('../sypp/src/lgem/'))
sys.path.insert(0, os.path.abspath('../'))


# from lgem.data import compute_embeddings, load_pseudobulk_data
from lgem.models import (
    LinearGeneExpressionModelLearned,
    LinearGeneExpressionModelOptimized,
)
from lgem.test import test as test_lgem
from lgem.train import train as train_lgem
from cris_test.single_norman_utils import separate_data, get_common_genes, compute_embeddings_double   

MODEL_DIR_PATH = '../cris_test/models/'
RESULTS_DIR_PATH = '../cris_test/results/'

# Parameters.
seed = 40 # prev 42
num_runs = 1
batch_size = 8
criterion = nn.MSELoss()
n_epochs = 25000
run_name = f"lgem_seed_{seed}_runs_{num_runs}"

 # PyTorch setup.
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Make results file path.
results_file_path = os.path.join(
    RESULTS_DIR_PATH, f"{run_name}_train_test_metrics.csv"
    )

# Import dataset
adata_filepath = '/wdir/tfm/SENA/data/Norman2019_raw.h5ad'
norman = sc.read(adata_filepath)
dataset_name = "norman"
prediction_type = "double" # Train has no test

# Get separate dataset for single, double perts and controls
norman_single, norman_double, norman_ctrl = separate_data(adata = norman, dataset_name = dataset_name)

if prediction_type == "double":
    # Join both AnnData datasets
    norman_both = norman_single.concatenate(norman_double, join = 'outer', index_unique = '-')
    all_perts, perts, genes, norman_common = get_common_genes(adata = norman_both, dataset_name = dataset_name)
else:
    # Y, perts and genes
    all_perts, perts, genes, norman_common = get_common_genes(adata_single = norman_single, dataset_name = dataset_name)

print(f"Number of unique perturbations: {len(perts)}/{len(all_perts)}")

# Pseudobulk the data per perturbation.
n_perts = len(perts)
n_genes = len(genes)
Y = torch.zeros((n_perts, n_genes), dtype=torch.float32)  # noqa: N806
for i, pert in tqdm(
    enumerate(perts), desc="Pseudobulking", total=n_perts, unit="perturbation"
):
    Y[i, :] = torch.Tensor(norman_common[norman_common.obs["condition_fixed"] == pert].X.mean(axis=0))

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

    if prediction_type == "double":
        num_runs = 1

    for current_run in range(num_runs):
        current_seed = seed + current_run # Redundant in double prediction
        torch.manual_seed(current_seed)
        model_name = f"lgem_seed_{current_seed}"

        # Directory
        savedir = os.path.join(MODEL_DIR_PATH, model_name)
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

        else:
            Y_train = sY
            P_train = sP
            full_dataset = TensorDataset(P_train, Y_train)
            doubles_idx = [i for i, pert in enumerate(perts) if "+" in pert]
            train_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

            doubles_dataset = TensorDataset(P[doubles_idx, :], Y[doubles_idx, :])
            test_dataloader = DataLoader(doubles_dataset, batch_size=batch_size, shuffle=False)
            # Save dataloaders as pickles and perturbation order from perts
            torch.save(train_dataloader, os.path.join(savedir, "train_dataloader.pt"))
            torch.save(test_dataloader, os.path.join(savedir, "test_dataloader.pt"))
            torch.save({"perts": perts}, os.path.join(savedir, "perts.pt"))
            
        # Fit the optimized model to the training data.
        model_optimized = LinearGeneExpressionModelOptimized(Y_train.T, G, P_train, b)
        train_loss_op = test_lgem(model_optimized, criterion, train_dataloader, device)
        print(f"Train loss (optimized model) | Run {current_run+1}: {train_loss_op:.4f}")
        test_loss_op = 0

        if prediction_type != "double":
            # Test the optimized model.
            test_loss_op = test_lgem(model_optimized, criterion, test_dataloader, device)
            print(f"Test loss (optimized model): {test_loss_op:.4f}")

        # Fit the learned model to the training data.
        model_learned = LinearGeneExpressionModelLearned(G, b)
        optimizer = torch.optim.Adam(params=model_learned.parameters(), lr=1e-3)
        model_learned = train_lgem(
            model_learned, criterion, optimizer, train_dataloader, n_epochs, device
        )
        train_loss_learn = test_lgem(model_optimized, criterion, train_dataloader, device)
        print(f"Train loss (optimized model) | Run {current_run+1}: {train_loss_op:.4f}")
        test_loss_learn = 0

        if prediction_type != "double":
            # Test the learned model.
            test_loss_learn = test_lgem(model_learned, criterion, test_dataloader, device)
            print(f"Test loss (learned model): {test_loss_learn:.4f}")  

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