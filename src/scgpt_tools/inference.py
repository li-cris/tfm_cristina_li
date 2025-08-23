import torch

from torch_geometric.loader import DataLoader

from typing import Dict, List, Optional
import random
from scipy.stats import pearsonr

from scgpt.model import TransformerGenerator

import numpy as np
from gears.utils import create_cell_graph_dataset_for_prediction
from gears import PertData

from data_utils.metrics import compute_kld, MMDLoss


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

                # Add perturbation flags to batch data
                actual_batch_size = batch_data.num_graphs
                pert_flags = [1 if gene in pert else 0 for gene in gene_list] # dim: (num_genes, )
                pert_flags = torch.tensor(pert_flags, dtype=torch.long, device=device)
                pert_flags = pert_flags.repeat(actual_batch_size) # (num_genes * eval_batch_size, )
                pert_flags = pert_flags.unsqueeze(1) # (num_genes * eval_batch_size, 1)

                assert (batch_data.x.size(0) == pert_flags.size(0)
                        ), f"Mismatch: x has {batch_data.x.size(0)}, pert_flags has {pert_flags.size(0)}"

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


class scGPTMetricEvaluator:
    """
    Class to evaluate predicted results in scGPT.
    For double predictions trained from single perturbation data;

    and single predictions trained from single perturbation data (Evaluates and sees if they are train/va/test)

    """

    def __init__(self, opts: None, gene_ids: np.ndarray, seed: int,
                 results_file_path: str, model: TransformerGenerator,
                 pertdata: PertData, evaluation_list: set = None):

        self.opts = opts
        self.gene_ids = gene_ids
        self.seed = seed
        self.results_file_path = results_file_path
        self.model = model
        self.pertdata = pertdata
        self.evaluation_list = evaluation_list

    def evaluate_double(self) -> dict:
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
        if self.evaluation_list is None:
            double_perturbations = set(
                [c for c in self.pertdata.adata.obs["condition"] if "ctrl" not in c]
            )
            print(f"Number of double perturbations: {len(double_perturbations)}")
        else:
            double_perturbations = self.evaluation_list

        pool_size = self.opts.pool_size
        eval_batch_size = self.opts.eval_batch_size
        include_zero_gene = self.opts.include_zero_gene
        top_deg = self.opts.top_deg

        with open(file=self.results_file_path, mode="w") as f:
            print(
                f"double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
                file=f,
            )

            print(f"Evaluating predictions with {pool_size} samples per double perturbation...")
            mean_result_pred = {}
            for i, double in enumerate(double_perturbations):

                print(f"Evaluating double {double}: {i + 1}/{len(double_perturbations)}")
                pert_list = [double.split("+")] # This should be a list, but it's only one double pert

                # Getting some samples for the current double pert
                double_adata = self.pertdata.adata[self.pertdata.adata.obs["condition"] == double]
                set_seeds(self.seed)
                if double_adata.n_obs>pool_size:
                    n = pool_size
                else:
                    # If less than pool size, keep only available samples
                    n = double_adata.n_obs
                    print(f"Not enough samples for double perturbation, using all available {n} samples.")
                random_indices = np.random.choice(double_adata.n_obs, size=n, replace=False)
                true_geps = double_adata[random_indices, :]

                # Get control samples
                # First control batch
                ctrl_adata = self.pertdata.adata[self.pertdata.adata.obs["condition"] == "ctrl"]
                random_indices = np.random.choice(
                    ctrl_adata.n_obs, size=n, replace=False
                    )
                ctrl_geps = ctrl_adata[random_indices, :]
                ctrl_geps_tensor = torch.tensor(ctrl_geps.X.toarray())
                # Second control batch
                random_indices = np.random.choice(
                    ctrl_adata.n_obs, size=n, replace=False
                )
                ctrl_geps_2 = ctrl_adata[random_indices, :]


                # Perturbation prediction for pool_size for current double
                result_pred = predict(model=self.model, pert_list=pert_list, pert_data=self.pert_data,
                                    eval_batch_size=eval_batch_size, include_zero_gene=include_zero_gene, gene_ids=self.gene_ids, pool_size=n)

                for pert in pert_list:
                    pert_name = "_".join(str(p) for p in pert)

                mean_result_pred[pert_name] = np.mean(result_pred, axis=0)


                # Tensor conversion and differential expression
                ctrl_ctrl_geps_tensor = torch.tensor(ctrl_geps_2.X.toarray()) - ctrl_geps_tensor
                true_ctrl_geps_tensor = torch.tensor(true_geps.X.toarray()) - ctrl_geps_tensor
                pred_ctrl_geps_tensor = torch.tensor(result_pred) - ctrl_geps_tensor


                # MMD
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

                ctrl_true_pearson_stat,  ctrl_true_pearson_pval = pearsonr(true_deg[topdeg_idx], ctrl_ctrl_deg[topdeg_idx])
                pred_true_pearson_stat,  pred_true_pearson_pval = pearsonr(true_deg[topdeg_idx], pred_deg[topdeg_idx])

                print(f"MMD (true vs. control):   {mmd_true_vs_ctrl:10.6f}")
                print(f"MMD (true vs. predicted): {mmd_true_vs_pred:10.6f}")
                print(f"MSE (true vs. control):   {mse_true_vs_ctrl:10.6f}")
                print(f"MSE (true vs. predicted): {mse_true_vs_pred:10.6f}")
                print(f"KLD (true vs. control):   {kld_true_vs_ctrl:10.6f}")
                print(f"KLD (true vs. predicted): {kld_true_vs_pred:10.6f}")
                print(f"Pearson Top {top_deg} DEG (true vs. control): {ctrl_true_pearson_stat:.6f} | p-value: {ctrl_true_pearson_pval:.6f}")
                print(f"Pearson Top {top_deg} DEG (true vs. predicted): {pred_true_pearson_stat:.6f} | p-value: {pred_true_pearson_pval:.6f}")


                print(
                    f"{double},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred},{kld_true_vs_ctrl},{kld_true_vs_pred},{ctrl_true_pearson_stat},{ctrl_true_pearson_pval},{pred_true_pearson_stat},{pred_true_pearson_pval}",
                    file=f,
                )

            print(f"Metrics saved to {self.results_file_path}.")

        return mean_result_pred


    def evaluate_single(self) -> dict:
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

        # 
        split_map = self.pertdata.adata.obs.set_index('condition')['split_dict'].to_dict()

        # If list of single perturbations is not given, evaluate some single perturbations.
        if self.evaluation_list is None:
            # Get all single perturbations.
            single_perturbations = set(
                [
                    c.strip("+ctrl")
                    for c in self.pertdata.adata.obs["condition"]
                    if ("ctrl+" in c) or ("+ctrl" in c)
                ]
            )
            print(f"Number of single perturbations: {len(single_perturbations)}")
            print("Reducing amount of single perturbations to predict to 250 random cases.")
            set_seeds(self.seed)
            single_perturbations = set(random.sample(single_perturbations, min(250, len(single_perturbations))))

        else:
            single_perturbations = self.evaluation_list

        pool_size = self.opts.pool_size
        eval_batch_size = self.opts.eval_batch_size
        include_zero_gene = self.opts.include_zero_gene
        top_deg = self.opts.top_deg

        with open(file=self.results_file_path, mode="w") as f:
            print(
                f"single,split1,split2,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
                file=f,
            )

            print(f"Evaluating predictions with {pool_size} samples per single perturbation...")
            mean_result_pred = {}
            for i, single in enumerate(single_perturbations):
                print(f"Evaluating single {i + 1}/{len(single_perturbations)}: {single}")
                split_type1 = split_map.get(f'{str(single)}+ctrl', 'Unknown')
                split_type2 = split_map.get(f'ctrl+{str(single)}', 'Unknown')

                # Get adata for single being evaluated
                true_geps = self.pertdata.adata[
                    (self.pertdata.adata.obs["condition"] == single) |
                    (self.pertdata.adata.obs["condition"] == '+'.join([single, 'ctrl'])) |
                    (self.pertdata.adata.obs["condition"] == '+'.join(['ctrl', single]))
                ]

                pert_list = [single.split("+")] # This should be a list, but it's only one double pert

                # Getting some samples for the current double pert
                set_seeds(self.seed)
                # Limiting n
                if true_geps.n_obs>pool_size:
                    n = pool_size
                    random_indices = np.random.choice(true_geps.n_obs, size=n, replace=False)
                    true_geps = true_geps[random_indices, :]
                else:
                    # If less than pool size, keep only available samples
                    n = true_geps.n_obs
                    print(f"Not enough samples for single perturbation, using all available {n} samples.")


                # Obtaining random sample of ctrl GEP
                all_ctrl_geps = self.pertdata.adata[self.pertdata.adata.obs["condition"] == "ctrl"]
                random_indices = np.random.choice(
                    all_ctrl_geps.n_obs, size=n, replace=False
                )
                ctrl_geps = all_ctrl_geps[random_indices, :]
                ctrl_geps_tensor = torch.tensor(ctrl_geps.X.toarray())

                # Another random ctrl_gep
                random_indices_2 = np.random.choice(
                    all_ctrl_geps.n_obs, size=n, replace=False
                )
                ctrl_geps_2 = all_ctrl_geps[random_indices_2, :]


                # Perturbation prediction for pool_size for current double
                result_pred = predict(model=self.model, pert_list=pert_list, pert_data=self.pertdata,
                                    eval_batch_size=eval_batch_size, include_zero_gene=include_zero_gene, gene_ids=self.gene_ids, pool_size=n)

                for pert in pert_list:
                    pert_name = "_".join(str(p) for p in pert)

                mean_result_pred[pert_name] = np.mean(result_pred, axis=0)


                # Tensor conversion and differential expression
                ctrl_ctrl_geps_tensor = torch.tensor(ctrl_geps_2.X.toarray()) - ctrl_geps_tensor
                true_ctrl_geps_tensor = torch.tensor(true_geps.X.toarray()) - ctrl_geps_tensor
                pred_ctrl_geps_tensor = torch.tensor(result_pred) - ctrl_geps_tensor


                # MMD
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

                ctrl_true_pearson_stat,  ctrl_true_pearson_pval = pearsonr(true_deg[topdeg_idx], ctrl_ctrl_deg[topdeg_idx])
                pred_true_pearson_stat,  pred_true_pearson_pval = pearsonr(true_deg[topdeg_idx], pred_deg[topdeg_idx])

                print(f"MMD (true vs. control):   {mmd_true_vs_ctrl:10.6f}")
                print(f"MMD (true vs. predicted): {mmd_true_vs_pred:10.6f}")
                print(f"MSE (true vs. control):   {mse_true_vs_ctrl:10.6f}")
                print(f"MSE (true vs. predicted): {mse_true_vs_pred:10.6f}")
                print(f"KLD (true vs. control):   {kld_true_vs_ctrl:10.6f}")
                print(f"KLD (true vs. predicted): {kld_true_vs_pred:10.6f}")
                print(f"Pearson Top {top_deg} DEG (true vs. control): {ctrl_true_pearson_stat:.6f} | p-value: {ctrl_true_pearson_pval:.6f}")
                print(f"Pearson Top {top_deg} DEG (true vs. predicted): {pred_true_pearson_stat:.6f} | p-value: {pred_true_pearson_pval:.6f}")


                print(
                    f"{single},{split_type1},{split_type2},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred},{kld_true_vs_ctrl},{kld_true_vs_pred},{ctrl_true_pearson_stat},{ctrl_true_pearson_pval},{pred_true_pearson_stat},{pred_true_pearson_pval}",
                    file=f,
                )

            print(f"Metrics saved to {self.results_file_path}.")

        return mean_result_pred