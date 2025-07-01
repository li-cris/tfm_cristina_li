import torch

from torch_geometric.loader import DataLoader

from typing import Dict, List, Optional

from scgpt_tool.model import TransformerGenerator

import numpy as np
from gears.utils import create_cell_graph_dataset_for_prediction
from gears import PertData

from data_utils.metrics import compute_kld, MMDLoss


def predict(
    model: TransformerGenerator, pert_list: List[str], pert_data: PertData,
    eval_batch_size: int,
    include_zero_gene: str,
    gene_ids: np.ndarray = None,
    amp: bool = True,
    pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.adata.var["gene_name"].values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )


    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for i, batch_data in enumerate(loader):
                batch_data.to(device)
                # create_cell_graph_dataset_for_prediction seems to return graph without pert flags
                if batch_data.x.dim() == 1:
                    batch_data.x = batch_data.x.unsqueeze(1) # (num_genes*eval_batch_size, 1)

                pert_flags = [1 if gene in pert else 0 for gene in gene_list] # dim: (num_genes, )
                pert_flags = torch.tensor(pert_flags, dtype=torch.long, device=device)
                pert_flags = pert_flags.repeat(eval_batch_size) # (num_genes * eval_batch_size, )
                pert_flags = pert_flags.unsqueeze(1) # (num_genes * eval_batch_size, 1)

                batch_data.x = torch.cat([batch_data.x, pert_flags], dim=1) # (num_genes*eval_batch_size, 2))


                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)

            preds = torch.cat(preds, dim=0)
            # results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
            # results_pred["_".join(pert)] = preds.detach().cpu().numpy()
            results_pred = preds.detach().cpu().numpy()
    return results_pred



def evaluate_double(opts: None,
                    gene_ids: np.ndarray,
                    double_results_file_path: str,
                    ctrl_geps_tensor: torch.Tensor,
                    model: TransformerGenerator,
                    pert_data: PertData,
                    double_perturbations: set = None) -> dict:
    """
    Evaluate the model's predictions on double perturbations and compute metrics.

    Given a trained perturbation model and a set of double perturbations (or all
    if none specified), this function predicts the gene expression changes induced
    by double perturbations and compares predictions against true experimental data.

    Metrics computed include:
    - Maximum Mean Discrepancy (MMD)
    - Mean Squared Error (MSE)
    - Kullback-Leibler Divergence (KLD)

    Results are logged to the specified output file and printed to the console.

    Args:
        opts (Options): Configuration object containing parameters like batch size,
                        pool size, random seed, and evaluation settings.
        gene_ids (np.ndarray): Array of gene indices mapping dataset genes to vocab tokens.
        double_results_file_path (str): File path to save evaluation metrics.
        ctrl_geps_tensor (torch.Tensor): Control gene expression tensor for comparison.
        model (TransformerGenerator): The trained perturbation prediction model.
        pert_data (PertData): Dataset object containing perturbation conditions and data.
        double_perturbations (set, optional): Set of double perturbation condition names
                                              to evaluate. If None, evaluates all doubles
                                              found in dataset excluding controls.

    Returns:
        dict: Mean predicted gene expression results per double perturbation.
    """


    # If list of double perturbations is not given, evaluate all doubles.
    if double_perturbations is None:
        double_perturbations = set(
            [c for c in pert_data.adata.obs["condition"] if "ctrl" not in c]
        )
        print(f"Number of double perturbations: {len(double_perturbations)}")

    pool_size = opts.pool_size
    eval_batch_size = opts.eval_batch_size
    include_zero_gene = opts.include_zero_gene
    seed = opts.seed

    with open(file=double_results_file_path, mode="w") as f:
        print(
            "double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred",
            file=f,
            )

        print(f"Evaluating predictions with {pool_size} samples pert double perturbation...")
        mean_result_pred = {}
        for i, double in enumerate(double_perturbations):

            print(f"Evaluating double {double}: {i + 1}/{len(double_perturbations)}")
            pert_list = [double.split("+")] # This should be a list, but it's only one double pert

            # Perturbation prediction for pool_size for current double
            result_pred = predict(model=model, pert_list=pert_list, pert_data=pert_data,
                                eval_batch_size=eval_batch_size, include_zero_gene=include_zero_gene, gene_ids=gene_ids, pool_size=pool_size)

            for pert in pert_list:
                pert_name = "_".join(str(p) for p in pert)

            mean_result_pred[pert_name] = np.mean(result_pred, axis=0)

            # Getting some samples for the current double pert
            np.random.seed(seed)
            double_adata = pert_data.adata[pert_data.adata.obs["condition"] == double]
            if double_adata.n_obs < pool_size:
                print(f"Warning: Not enough samples for {double}. Randomly selection samples from {double_adata.n_obs} samples.")
                true_geps = double_adata.copy()
                true_geps = true_geps[np.random.choice(double_adata.n_obs, size=pool_size, replace=True), :]

            else:
                random_indices = np.random.choice(double_adata.n_obs, size=pool_size, replace=False)
                true_geps = double_adata[random_indices, :]

            # Tensorize data
            # pred_geps_tensor = torch.tensor(result_pred["_".join(pert_list)])
            pred_geps_tensor = torch.tensor(result_pred)
            true_geps_tensor = torch.tensor(true_geps.X.toarray())


            # MMD
            # MMD setup.
            mmd_sigma = 200.0
            kernel_num = 10
            mmd_loss = MMDLoss(fix_sigma=mmd_sigma, kernel_num=kernel_num)
            # Compute MMD for the entire batch.
            mmd_true_vs_ctrl = mmd_loss.forward(
                source=ctrl_geps_tensor, target=true_geps_tensor
                )

            mmd_true_vs_pred = mmd_loss.forward(
                source=pred_geps_tensor, target=true_geps_tensor
                )


            # Compute MSE for the entire batch.
            mse_true_vs_ctrl = torch.mean(
                (true_geps_tensor - ctrl_geps_tensor) ** 2
                ).item()

            mse_true_vs_pred = torch.mean(
                (true_geps_tensor - pred_geps_tensor) ** 2
                ).item()

            # Compute KLD
            kld_true_vs_ctrl = compute_kld(true_geps_tensor, ctrl_geps_tensor)
            kld_true_vs_pred = compute_kld(true_geps_tensor, pred_geps_tensor)

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

        print(f"Metrics saved to {double_results_file_path}.")

    return mean_result_pred