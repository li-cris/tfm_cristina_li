from typing import Dict, Tuple
import torch
import torch.nn as nn

import git
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from anndata import AnnData
from scipy.sparse import csr_matrix
from tqdm import tqdm
import json
import random

from scipy.stats import pearsonr
from data_utils.metrics import MMDLoss, compute_kld

def get_git_root() -> str:
    """Get the root directory of the Git repository."""
    return git.Repo(search_parent_directories=True).working_tree_dir

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

            mse_loss_list.extend(mse_loss.mean(dim=1).cpu().numpy())


    double_predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return double_perts_list, double_predictions, ground_truth, mse_loss_list



def evaluate_double_metrics(double_adata: AnnData, ctrl_adata: AnnData, model_name: str,
                            predictions: np.ndarray,
                            results_savedir: str,
                            double_perts: list,
                            pool_size: int = 200, seed: int = 42, top_deg: int = 20,
                            model_type: str = 'op') -> None:
    """Evaluate metrics for double perturbtaions."""

    # Make results file path.

    results_file_path = os.path.join(
        results_savedir, f"{model_name}_double_metrics_{model_type}.csv"
    )

    with open(file=results_file_path, mode="w") as f:
        print(
            f"double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
            file=f,
        )
        for i, double in enumerate(double_perts):
            print(f"Evaluating double {i + 1}/{len(double_perts)}: {double}")

            # Single prediction from the lgem model
            pred_geps = predictions[i, :]
            pred_geps = np.array([pred_geps])

            # Take all or pool_size perturbation samples for pert
            true_geps = double_adata[double_adata.obs['condition_fixed'] == double].copy()
            # Limiting n
            if true_geps.n_obs>pool_size:
                n = pool_size
                random_indices = np.random.choice(true_geps.n_obs, size=n, replace=False)
                true_geps = true_geps[random_indices, :]  
            else:
                # If less than pool size, keep only available samples
                n = true_geps.n_obs
                print(f"Not enough samples for double perturbation, using all available {n} samples.")

            # Radomly take two sets of control cells
            # Obtaining random sample of ctrl GEP
            # Control saved in pertdata_ctrl (AnnData object)
            set_seeds(seed)
            random_indices = np.random.choice(
                ctrl_adata.n_obs, size=n, replace=False
            )
            ctrl_geps = ctrl_adata[random_indices, :]
            pred_geps = csr_matrix(np.tile(np.array(pred_geps), reps=(n, 1)))

            # Another random ctrl_gep
            random_indices_2 = np.random.choice(
                ctrl_adata.n_obs, size=n, replace=False
            )
            ctrl_geps_2 = ctrl_adata[random_indices_2, :]

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

            # Compute Pearson for top DEG
            true_deg = true_ctrl_geps_tensor.mean(dim=0).cpu().detach().numpy()
            ctrl_ctrl_deg = ctrl_ctrl_geps_tensor.mean(dim=0).cpu().detach().numpy()
            pred_deg = pred_ctrl_geps_tensor.mean(dim=0).cpu().detach().numpy()
            topdeg_idx = np.argsort(abs(true_deg))[-top_deg:]

            pearson_true_vs_ctrl = pearsonr(true_deg[topdeg_idx], ctrl_ctrl_deg[topdeg_idx])
            pearson_true_vs_pred = pearsonr(true_deg[topdeg_idx], pred_deg[topdeg_idx])

            print(f"MMD (true vs. control):   {mmd_true_vs_ctrl:10.6f}")
            print(f"MMD (true vs. predicted): {mmd_true_vs_pred:10.6f}")
            print(f"MSE (true vs. control):   {mse_true_vs_ctrl:10.6f}")
            print(f"MSE (true vs. predicted): {mse_true_vs_pred:10.6f}")
            print(f"KLD (true vs. control):   {kld_true_vs_ctrl:10.6f}")
            print(f"KLD (true vs. predicted): {kld_true_vs_pred:10.6f}")
            print(f"Pearson Top {top_deg} DEG (true vs. control): {pearson_true_vs_ctrl.statistic:.6f} | p-value: {pearson_true_vs_ctrl.pvalue:.6f}")
            print(f"Pearson Top {top_deg} DEG (true vs. predicted): {pearson_true_vs_pred.statistic:.6f} | p-value: {pearson_true_vs_pred.pvalue:.6f}")

            print(
                f"{double},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred},{kld_true_vs_ctrl},{kld_true_vs_pred},{pearson_true_vs_ctrl.statistic},{pearson_true_vs_ctrl.pvalue},{pearson_true_vs_pred.statistic},{pearson_true_vs_pred.pvalue}",
                file=f,
            )
        print(f"Results saved to {results_file_path}.")