import os

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import random

from data_utils.metrics import MMDLoss, compute_kld


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_double(adata: AnnData, model_name: str, results_savedir: str,
                    pool_size: int = 250, seed: int = 42, top_deg: int = 20) -> None:
    """Evaluate the predicted GEPs of double perturbations."""
    # Load predicted GEPs.
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(results_savedir, f"{model_name}_double.csv")
    )

    # Make results file path.
    results_file_path = os.path.join(
        results_savedir, f"{model_name}_double_metrics.csv"
    )

    with open(file=results_file_path, mode="w") as f:
        print(
            f"double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
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
                # If less than pool size, keep only available samples
                n = true_geps.n_obs
                print(f"Not enough samples for double perturbation, using all available {n} samples.")

            set_seeds(seed)

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


def evaluate_single(adata: AnnData, model_name: str, results_savedir: str,
                    pool_size: int = 250, seed: int = 42, top_deg: int = 20) -> None:
    """Evaluate the predicted GEPs of single perturbations."""

    # Mapping of data type during train/val/test
    split_map = adata.obs.set_index('condition')['split'].to_dict()

    # Load predicted GEPs.
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(results_savedir, f"{model_name}_single.csv")
    )

    # Make results file path.
    results_file_path = os.path.join(
        results_savedir, f"{model_name}_single_metrics.csv"
    )

    with open(file=results_file_path, mode="w") as f:
        print(
            f"single,split1,split2,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
            file=f,
        )

        for i, single in enumerate(df["single"]):
            # Get the predicted GEP for the current single perturbation.
            pred_geps = df.loc[df["single"] == single]
            pred_geps = pred_geps.iloc[0, 1:].tolist()
            pred_geps = np.array([pred_geps])

            # Get split during train/val/test
            split_type1 = split_map.get('+'.join([single, 'ctrl']), 'Unknown')
            split_type2 = split_map.get('+'.join(['ctrl', single]), 'Unknown')

            # Get all the true GEPs with the current single perturbation.
            print(f"Evaluating single {i + 1}/{len(df['single'])}: {single}")
            true_geps = adata[
                (adata.obs["condition"] == single) |
                (adata.obs["condition"] == '+'.join([single, 'ctrl'])) |
                (adata.obs["condition"] == '+'.join(['ctrl', single]))
            ]

            # Limiting n
            if true_geps.n_obs>pool_size:
                n = pool_size
                random_indices = np.random.choice(true_geps.n_obs, size=n, replace=False)
                true_geps = true_geps[random_indices, :]
            else:
                # If less than pool size, keep only available samples
                n = true_geps.n_obs
                print(f"Not enough samples for single perturbation, using all available {n} samples.")

            set_seeds(seed)

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
                f"{single},{split_type1},{split_type2},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred},{kld_true_vs_ctrl},{kld_true_vs_pred},{pearson_true_vs_ctrl.statistic},{pearson_true_vs_ctrl.pvalue},{pearson_true_vs_pred.statistic},{pearson_true_vs_pred.pvalue}",
                file=f,
            )