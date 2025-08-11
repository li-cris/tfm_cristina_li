
import numpy as np
import pandas as pd
import torch
import random
import os
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

from data_utils.metrics import MMDLoss, compute_kld
from sena.sena_utils import find_pert_pairs


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_double(model,
                    dataloader,
                    data_path,
                    ptb_genes,
                    config,
                    results_dir_path: str,
                    device: str,
                    run_name: str,
                    numint: int = 2) -> None:
    """
    W
    """
    # Load config
    pool_size = config.get("pool_size", 150)
    MMD_sigma = config.get("MMD_sigma", 200.0)
    kernel_num = config.get("kernel_num", 10)
    # matched_IO = config.get("matched_IO", False)
    temp = config.get("temp", 1000.0)
    seed = config.get("seed", 42)
    top_deg = config.get("top_deg", 100)
    # latdim = config.get("latdim", 105)
    # model_name = config.get("name", "example")
    batch_size = 5

    set_seeds(seed)

    # Load raw data, Norman for now
    if os.path.exists(data_path):
        adata = sc.read_h5ad(data_path)
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")


    # Only keeps the matrix
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"].X.copy()
    baseline_ctrl = np.mean(ctrl_adata.toarray(), axis=0)

    # Load double perturbation data
    ptb_targets = sorted(adata.obs["guide_ids"].unique().tolist())[1:]
    double_adata = adata[
        (adata.obs["guide_ids"].str.contains(","))
        & (
            adata.obs["guide_ids"].map(
                lambda x: all([y in ptb_targets for y in x.split(",")])
            )
        )
    ]

    # Randomply sampling same subset of control data based on seed
    ctrl_random = ctrl_adata[np.random.choice(ctrl_adata.shape[0], pool_size, replace=True)]

    print("Proceeding with evaluation of intervened pairs.")
    # Make results file path.
    results_file_path = os.path.join(
        results_dir_path, f"{run_name}_double_metrics.csv"
        )

    # Finding perturbation pairs for model and shape of c (for each dataloader)
    all_pairs, _, c_shape = find_pert_pairs(dataloader=dataloader, device=device)

    # Prepare writing results to file
    with open(file=results_file_path, mode="w") as f:
            print(
                f"double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_ctrl,Pearson_pval_true_vs_ctrl,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
                file=f,
            )

            # Initialise lists  
            pred_x_list, gt_x_list = [], []
            gt_y_list, pred_y_list = [], []
            c_y_list, mu_list, var_list = [], [], []
            MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
            MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []
            full_preds = []
            predictions_list = [] # initialise dataframe to save mean predictions for each double

            # Initialise MMD function
            mmd_loss_func = MMDLoss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

            for num, unique_pairs in enumerate(all_pairs):
                gene_pair = "+".join([ptb_genes[unique_pairs[0]], ptb_genes[unique_pairs[1]]])
                print(f"Evaluating for intervention pair {num+1}/{len(all_pairs)}: {gene_pair}")

                # Turning control samples into tensor
                ctrl_geps_tensor = torch.tensor(ctrl_random.toarray()).double()
                # ctrl_geps_tensor = torch.tensor(ctrl_random.toarray())

                # Get equivalent double adata or random sample
                unique_pairs_set = set([ptb_genes[unique_pairs[0]], ptb_genes[unique_pairs[1]]])
                double_samples = double_adata[double_adata.obs["guide_ids"].map(
                     lambda x: all([y in unique_pairs_set for y in x.split(",")])
                )]

                if double_samples.n_obs == 0:
                    print(f"0 samples found for {gene_pair}. Taking random samples from whole adata.")
                    true_geps = adata.copy()
                    random_indices = np.random.choice(
                        adata.n_obs, size=pool_size, replace=True
                        )
                    true_geps = true_geps[random_indices, :]

                elif double_samples.n_obs < pool_size:
                    print(f"Warning: Not enough samples for {gene_pair}. Randomly selecting samples from {double_samples.n_obs} samples.")
                    true_geps = double_samples.copy()
                    true_geps = true_geps[np.random.choice(double_samples.n_obs, size=pool_size, replace=True), :]

                else:
                    random_indices = np.random.choice(double_samples.n_obs, size=pool_size, replace=False)
                    true_geps = double_samples[random_indices, :]

                # Turning true samples from predictions into tensor
                baseline_true = np.mean(true_geps.X.toarray(), axis=0)
                true_geps_tensor = torch.tensor(true_geps.X.toarray()).double()

                # Saving both in same Tensor Dataset
                ctrl_double_dataset = TensorDataset(ctrl_geps_tensor, true_geps_tensor)
                ctrl_double_loader = DataLoader(ctrl_double_dataset, batch_size=32, shuffle=False)

                for i, (x, y) in enumerate(ctrl_double_loader):
                    x = x.to(device)
                    y = y.to(device)

                    c_shape_loader = c_shape[0, :].repeat(32*batch_size, 1)
                    c1 = torch.zeros_like(c_shape_loader).double().to(device)
                    c1[:, unique_pairs[0]] = 1
                    c2 = torch.zeros_like(c_shape_loader).double().to(device)
                    c2[:, unique_pairs[1]] = 1

                    if len(x) < len(c1):
                        c1 = c1[:len(x), :]
                        c2 = c2[:len(x), :]

                    with torch.no_grad():
                        y_hat, x_recon, z_mu, z_var, _, _ = model(
                            x, c1, c2, num_interv=numint, temp=temp)

                    gt_x_list.append(x.cpu())
                    pred_x_list.append(x_recon.cpu())

                    gt_y_list.append(y.cpu())
                    pred_y_list.append(y_hat.cpu())
                    full_preds.append(y_hat.cpu())

                    c_y_list.append(c_shape.cpu())
                    mu_list.append(z_mu.cpu())
                    var_list.append(z_var.cpu())


                    # Limit stacked tensors while iterating through desired indices
                    if len(gt_x_list) >= batch_size:

                        (gt_x_list, pred_x_list, gt_y_list,
                        pred_y_list, c_y_list, mu_list, var_list,
                        MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                        MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp) = compute_metrics(gt_x_list, pred_x_list, gt_y_list,
                                                                                                    pred_y_list, c_y_list, mu_list, var_list,
                                                                                                    MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                                                                                                    MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp,
                                                                                                    mmd_loss_func)

                        # Reset lists
                        pred_x_list, gt_x_list = [], []
                        gt_y_list, pred_y_list = [], []
                        c_y_list, mu_list, var_list = [], [], []

                # Once iterated through all indices, check rest of stacked tensors
                if len(gt_x_list) > 0:

                    (gt_x_list, pred_x_list, gt_y_list,
                    pred_y_list, c_y_list, mu_list, var_list,
                    MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                    MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp) = compute_metrics(gt_x_list, pred_x_list, gt_y_list,
                                                                                                pred_y_list, c_y_list, mu_list, var_list,
                                                                                                MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                                                                                                MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp,
                                                                                                mmd_loss_func)


                # After each unique pair, print and save metrics
                print(f"MMD (true vs. control):     {np.mean(MMD_ctrl_true_temp):.6f}")
                print(f"MMD (true vs. predicted):   {np.mean(MMD_pred_true_temp):.6f}")
                print(f"MSE (true vs. control):     {np.mean(MSE_ctrl_true_temp):.6f}")
                print(f"MSE (true vs. predicted):   {np.mean(MSE_pred_true_temp):.6f}")
                print(f"KLD (true vs. control):     {np.mean(KLD_ctrl_true_temp):.6f}")
                print(f"KLD (true vs. predicted):   {np.mean(KLD_pred_true_temp):.6f}")

                # Calculating Pearson with mean GEPs
                mean_pred = torch.vstack(full_preds).mean(dim=0).cpu()
                true_deg = baseline_true-baseline_ctrl
                pred_deg = mean_pred.numpy()-baseline_ctrl
                ctrl_ctrl_deg = baseline_ctrl-baseline_ctrl
                deg_idx = np.argsort(abs(true_deg))[-top_deg:] # DEG from true pert value

                pred_true_pearson_stat,  pred_true_pearson_pval = pearsonr(true_deg[deg_idx], pred_deg[deg_idx])
                ctrl_true_pearson_stat,  ctrl_true_pearson_pval = pearsonr(true_deg[deg_idx], ctrl_ctrl_deg[deg_idx])

                print(f"Pearson Top {top_deg} DEG (true vs. control): {ctrl_true_pearson_stat:.6f} | p-value: {ctrl_true_pearson_pval:.6f}")
                print(f"Pearson Top {top_deg} DEG (true vs. predicted): {pred_true_pearson_stat:.6f} | p-value: {pred_true_pearson_pval:.6f}")

                # Save results to file
                print(f"{gene_pair},{np.mean(MMD_ctrl_true_temp)},{np.mean(MMD_pred_true_temp)},{np.mean(MSE_ctrl_true_temp)},{np.mean(MSE_pred_true_temp)},{np.mean(KLD_ctrl_true_temp)},{np.mean(KLD_pred_true_temp)},{ctrl_true_pearson_stat},{ctrl_true_pearson_pval},{pred_true_pearson_stat},{pred_true_pearson_pval}",
                    file=f,
                )

                # Saving and updating predictions
                prediction_row = [gene_pair] + mean_pred.tolist()
                predictions_list.append(prediction_row)


                # Reset lists
                pred_x_list, gt_x_list = [], []
                gt_y_list, pred_y_list = [], []
                c_y_list, mu_list, var_list = [], [], []
                MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
                MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []


            # Save prediction data
            prediction_colnames = ['double'] + adata.var_names.tolist()
            prediction_df = pd.DataFrame(predictions_list, columns=prediction_colnames)
            prediction_file_path = os.path.join(results_dir_path, f"{run_name}_double_prediction.csv")
            prediction_df.to_csv(prediction_file_path)
            print(f"Saved predictions at {prediction_file_path}")


def evaluate_single(model,
                    dataloader,
                    data_path,
                    ptb_genes,
                    config,
                    results_dir_path: str,
                    device: str,
                    run_name: str,
                    numint: int = 1,
                    split_ptb = None) -> None:
    """
    W
    """
    # Load config
    pool_size = config.get("pool_size", 150)
    MMD_sigma = config.get("MMD_sigma", 200.0)
    kernel_num = config.get("kernel_num", 10)
    # matched_IO = config.get("matched_IO", False)
    temp = config.get("temp", 1000.0)
    seed = config.get("seed", 42)
    top_deg = config.get("top_deg", 100)
    # latdim = config.get("latdim", 105)
    # model_name = config.get("name", "example")
    batch_size = 5
    
    if split_ptb is not None:
        split_ptbs = split_ptb
    else:
        split_ptbs=[
            "ETS2",
            "SGK1",
            "POU3F2",
            "TBX2",
            "CBL",
            "MAPK1",
            "CDKN1C",
            "S1PR2",
            "PTPN1",
            "MAP2K6",
            "COL1A1",
        ]

    # Load raw data, Norman for now
    if os.path.exists(data_path):
        adata = sc.read_h5ad(data_path)
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")


    # Only keeps the matrix
    set_seeds(seed)
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"].X.copy()
    baseline_ctrl = np.mean(ctrl_adata.toarray(), axis=0)

    # Load single perturbation data
    single_adata = adata[
        (~adata.obs["guide_ids"].str.contains(","))
        & (adata.obs["guide_ids"] != "")
        ].copy()

    # Randomply sampling same subset of control data based on seed
    ctrl_random = ctrl_adata[np.random.choice(ctrl_adata.shape[0], pool_size, replace=True)]

    print("Proceeding with evaluation of intervened genes.")
    # Make results file path.
    results_file_path = os.path.join(
        results_dir_path, f"{run_name}_single_metrics.csv"
        )

    # Finding perturbation pairs for model and shape of c (for each dataloader) Review
    all_pairs, _, c_shape = find_pert_pairs(dataloader=dataloader, device=device)

    # Prepare writing results to file
    with open(file=results_file_path, mode="w") as f:
            print(
                f"single,split,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred,PearsonTop{top_deg}_true_vs_pred,Pearson_pval_true_vs_pred",
                file=f,
            )

            # Initialise lists  
            pred_x_list, gt_x_list = [], []
            gt_y_list, pred_y_list = [], []
            c_y_list, mu_list, var_list = [], [], []
            MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
            MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []
            full_preds = []
            predictions_list = [] # initialise dataframe to save mean predictions for each double

            # Initialise MMD function
            mmd_loss_func = MMDLoss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

            if len(ptb_genes)>250:
                # Keep 300 random ptb_genes
                print("Predicting for random 250 perturbation targets.")
                ptb_genes = np.array(random.sample(list(ptb_genes), 250))

            for num, unique_pert in enumerate(ptb_genes):
                print(f"Evaluating for intervention pair {num+1}/{len(ptb_genes)}: {unique_pert}")

                # Turning control samples into tensor
                ctrl_geps_tensor = torch.tensor(ctrl_random.toarray()).double()
                # ctrl_geps_tensor = torch.tensor(ctrl_random.toarray())

                # Get equivalent single adata or random sample
                single_samples = single_adata[single_adata.obs["guide_ids"] == unique_pert]

                if single_samples.n_obs == 0:
                    print(f"0 samples found for {unique_pert}. Taking random samples from whole adata.")
                    true_geps = adata.copy()
                    set_seeds(seed)
                    random_indices = np.random.choice(
                        adata.n_obs, size=pool_size, replace=True
                        )
                    true_geps = true_geps[random_indices, :]

                elif single_samples.n_obs < pool_size:
                    print(f"Warning: Not enough samples for {unique_pert}. Keeping all samples from {single_samples.n_obs} samples.")
                    true_geps = single_samples.copy()

                    set_seeds(seed)
                    n = single_samples.n_obs
                    ctrl_geps = ctrl_random[np.random.choice(pool_size, size=n, replace=False), :] # Instead of n_obs, pool_size because random set of control samples always is n=pool_size
                    ctrl_geps_tensor = torch.tensor(ctrl_geps.toarray()).double()

                else:
                    # Takes pool_size number of samples from single_samples
                    set_seeds(seed)
                    random_indices = np.random.choice(single_samples.n_obs, size=pool_size, replace=False)
                    true_geps = single_samples[random_indices, :]

                # Turning true samples from predictions into tensor
                baseline_true = np.mean(true_geps.X.toarray(), axis=0)
                true_geps_tensor = torch.tensor(true_geps.X.toarray()).double()

                # Saving both in same Tensor Dataset
                ctrl_single_dataset = TensorDataset(ctrl_geps_tensor, true_geps_tensor)
                ctrl_single_loader = DataLoader(ctrl_single_dataset, batch_size=32, shuffle=False)

                for i, (x, y) in enumerate(ctrl_single_loader):
                    x = x.to(device)
                    y = y.to(device)

                    c_shape_loader = c_shape[0, :].repeat(32*batch_size, 1)
                    c = torch.zeros_like(c_shape_loader).double().to(device)
                    c[:, num] = 1

                    if len(x) < len(c):
                        c = c[:len(x), :]

                    with torch.no_grad():
                        y_hat, x_recon, z_mu, z_var, _, _ = model(
                            x, c, c, num_interv=numint, temp=temp)

                    gt_x_list.append(x.cpu())
                    pred_x_list.append(x_recon.cpu())

                    gt_y_list.append(y.cpu())
                    pred_y_list.append(y_hat.cpu())
                    full_preds.append(y_hat.cpu())

                    c_y_list.append(c_shape.cpu())
                    mu_list.append(z_mu.cpu())
                    var_list.append(z_var.cpu())


                    # Limit stacked tensors while iterating through desired indices
                    if len(gt_x_list) >= batch_size:

                        (gt_x_list, pred_x_list, gt_y_list,
                        pred_y_list, c_y_list, mu_list, var_list,
                        MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                        MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp) = compute_metrics(gt_x_list, pred_x_list, gt_y_list,
                                                                                                    pred_y_list, c_y_list, mu_list, var_list,
                                                                                                    MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                                                                                                    MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp,
                                                                                                    mmd_loss_func)

                        # Reset lists
                        pred_x_list, gt_x_list = [], []
                        gt_y_list, pred_y_list = [], []
                        c_y_list, mu_list, var_list = [], [], []

                # Once iterated through all indices, check rest of stacked tensors
                if len(gt_x_list) > 0:

                    (gt_x_list, pred_x_list, gt_y_list,
                    pred_y_list, c_y_list, mu_list, var_list,
                    MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                    MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp) = compute_metrics(gt_x_list, pred_x_list, gt_y_list,
                                                                                                pred_y_list, c_y_list, mu_list, var_list,
                                                                                                MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                                                                                                MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp,
                                                                                                mmd_loss_func)


                # After each unique pair, print and save metrics
                print(f"MMD (true vs. control):     {np.mean(MMD_ctrl_true_temp):.6f}")
                print(f"MMD (true vs. predicted):   {np.mean(MMD_pred_true_temp):.6f}")
                print(f"MSE (true vs. control):     {np.mean(MSE_ctrl_true_temp):.6f}")
                print(f"MSE (true vs. predicted):   {np.mean(MSE_pred_true_temp):.6f}")
                print(f"KLD (true vs. control):     {np.mean(KLD_ctrl_true_temp):.6f}")
                print(f"KLD (true vs. predicted):   {np.mean(KLD_pred_true_temp):.6f}")

                # Calculating Pearson with mean GEPs
                mean_pred = torch.vstack(full_preds).mean(dim=0).cpu()
                true_deg = baseline_true-baseline_ctrl
                pred_deg = mean_pred.numpy()-baseline_ctrl
                deg_idx = np.argsort(abs(true_deg))[-top_deg:] # DEG from true pert value

                pred_true_pearson_stat,  pred_true_pearson_pval = pearsonr(true_deg[deg_idx], pred_deg[deg_idx])

                print(f"Pearson Top {top_deg} DEG (true vs. predicted): {pred_true_pearson_stat:.6f} | p-value: {pred_true_pearson_pval:.6f}")

                if str(unique_pert) in split_ptbs:
                    split_type = "test"
                else:
                    split_type = "train"

                # Save results to file
                print(f"{unique_pert},{split_type},{np.mean(MMD_ctrl_true_temp)},{np.mean(MMD_pred_true_temp)},{np.mean(MSE_ctrl_true_temp)},{np.mean(MSE_pred_true_temp)},{np.mean(KLD_ctrl_true_temp)},{np.mean(KLD_pred_true_temp)},{pred_true_pearson_stat},{pred_true_pearson_pval}",
                    file=f,
                )

                # Saving and updating predictions
                prediction_row = [unique_pert] + mean_pred.tolist()
                predictions_list.append(prediction_row)


                # Reset lists
                pred_x_list, gt_x_list = [], []
                gt_y_list, pred_y_list = [], []
                c_y_list, mu_list, var_list = [], [], []
                MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
                MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []


            # Save prediction data
            prediction_colnames = ['single'] + adata.var_names.tolist()
            prediction_df = pd.DataFrame(predictions_list, columns=prediction_colnames)
            prediction_file_path = os.path.join(results_dir_path, f"{run_name}_single_prediction.csv")
            prediction_df.to_csv(prediction_file_path)
            print(f"Saved predictions at {prediction_file_path}")


# Still need to save predictions somewhere (forgot)

def compute_metrics(gt_x_list, pred_x_list, gt_y_list,
                    pred_y_list, c_y_list, mu_list, var_list,
                    MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
                    MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp,
                    mmd_loss_func):
    """
    asdf
    """
    gt_x = torch.vstack(gt_x_list)
    pred_x = torch.vstack(pred_x_list)
    gt_y = torch.vstack(gt_y_list)
    pred_y = torch.vstack(pred_y_list)
    c_y = torch.vstack(c_y_list)
    mu = torch.vstack(mu_list)
    var = torch.vstack(var_list)

    # Compute MSE
    MSE_pred_true = torch.mean((gt_y - pred_y)**2)
    MSE_ctrl_true = torch.mean((gt_y - gt_x)**2)

    # Compute MMD
    MMD_pred_true = mmd_loss_func(gt_y, pred_y)
    MMD_ctrl_true = mmd_loss_func(gt_y, gt_x)

    # Compute KLD
    KLD_pred_true = compute_kld(gt_y, pred_y)
    KLD_ctrl_true = compute_kld(gt_y, gt_x)


    # Save temporal calculations
    MSE_pred_true_temp.append(MSE_pred_true.item())
    MSE_ctrl_true_temp.append(MSE_ctrl_true.item())

    MMD_pred_true_temp.append(MMD_pred_true.item())
    MMD_ctrl_true_temp.append(MMD_ctrl_true.item())

    KLD_pred_true_temp.append(KLD_pred_true.item())
    KLD_ctrl_true_temp.append(KLD_ctrl_true.item())

    return (gt_x_list, pred_x_list, gt_y_list,
            pred_y_list, c_y_list, mu_list, var_list,
            MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp,
            MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp)