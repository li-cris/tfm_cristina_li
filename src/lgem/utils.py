from typing import Dict, Tuple
import torch
import torch.nn as nn

import git
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_git_root() -> str:
    """Get the root directory of the Git repository."""
    return git.Repo(search_parent_directories=True).working_tree_dir


def load_gene_pathway_mask(
    map_file_path: str,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Load gene-pathway mask from file.

    The pathway-gene map file must be a tab-separated file with two columns:
    pathway_id and ensembl_gene_id.

    Args:
        map_file_path: Path to pathway-gene map file.

    Returns:
        mask: Gene-pathway mask with shape (n_genes, n_pathways).
        gene_to_index: Mapping from gene to index.
        pathway_to_index: Mapping from pathway to index.
    """
    # Load the pathway-gene map file.
    df = pd.read_csv(map_file_path, sep="\t")

    # Strip any leading or trailing whitespace from the columns.
    df["pathway_id"] = df["pathway_id"].str.strip()
    df["ensembl_gene_id"] = df["ensembl_gene_id"].str.strip()

    # Get unique genes and pathways.
    genes = df["ensembl_gene_id"].unique()
    pathways = df["pathway_id"].unique()

    # Create mappings: gene -> index, pathway -> index.
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    pathway_to_index = {pathway: idx for idx, pathway in enumerate(pathways)}

    # Initialize the mask matrix.
    mask = np.zeros((len(genes), len(pathways)), dtype=int)

    # Fill the mask matrix.
    for _, row in tqdm(
        df.iterrows(), desc="Loading gene-pathway mask", total=len(df), unit="pathway"
    ):
        gene_idx = gene_to_index[row["ensembl_gene_id"]]
        pathway_idx = pathway_to_index[row["pathway_id"]]
        mask[gene_idx, pathway_idx] = 1

    return mask, gene_to_index, pathway_to_index

# evaluate model
def predict_lgem_singles(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_P, _ in dataloader:  # noqa: N806
            batch_P = batch_P.to(device)  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            predictions.append(Y_predicted.cpu().numpy())

    single_predictions = np.concatenate(predictions, axis=1)
    return single_predictions

def predict_evaluate_lgem_double(model, device, dataloader, perts_list):
    """Predicts the double prediction output of the model based on embedding of double perturbations."""
    model.to(device)
    model.eval()
    double_perts_list = [pert for pert in perts_list if "+" in pert]
    predictions = []
    ground_truth = [] 
    mse_loss_list = []
    mse_loss_fn = nn.MSELoss(reduction = "none")
    with torch.no_grad():
        print("Predicting and calculating loss for double perturbations.")
        for batch_P, batch_Y in dataloader:  # noqa: N806
            batch_P, batch_Y = batch_P.to(device), batch_Y.to(device)  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            mse_loss = mse_loss_fn(Y_predicted.T, batch_Y)
            predictions.append(Y_predicted.T.cpu().numpy())
            ground_truth.append(batch_Y.cpu().numpy())

            mse_loss_list.extend(mse_loss.cpu().numpy())


    double_predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return double_perts_list, double_predictions, ground_truth, mse_loss_list
