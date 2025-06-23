import os
import torch
import numpy as np
from tqdm import tqdm

from data_utils.metrics import MMDLoss, compute_kld
from sena.utils import find_pert_pairs

def evaluate_model(model, dataloader, ptb_genes, config, results_dir_path: str, device: str, run_name: str, numint: int = 2) -> None:
    """Evaluate the model on the dataloader perturbation by perturbation and save the results."""

    # Finding unique combinations and index equivalent in whole data
    all_pairs, pair_indices, c_shape = find_pert_pairs(dataloader=dataloader, device=device)

    # Load config
    MMD_sigma = config.get("MMD_sigma", 200.0)
    kernel_num = config.get("kernel_num", 10)
    matched_IO = config.get("matched_IO", False)
    temp = config.get("temp", 1000.0)
    seed = config.get("seed", 42)
    latdim = config.get("latdim", 105)
    model_name = config.get("name", "example")
    batch_size = 10 # batch size for evaluation dataloader

    print("Proceeding with evaluation of intervened pairs.")
    # Make results file path.
    results_file_path = os.path.join(
        results_dir_path, f"{run_name}_double_metrics.csv"
        )

    # Prepare writing results to file
    with open(file=results_file_path, mode="w") as f:
            print(
                "double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred",
                file=f,
            )

            # Initialise lists  
            pred_x_list, gt_x_list = [], []
            gt_y_list, pred_y_list = [], []
            c_y_list, mu_list, var_list = [], [], []
            MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
            MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []
            MSE_pred_true_l, KLD_pred_true_l, MMD_pred_true_l = [], [], []
            MSE_ctrl_true_l, KLD_ctrl_true_l, MMD_ctrl_true_l = [], [], []

            # Initialise MMD function
            mmd_loss_func = MMDLoss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

            for num, unique_pairs in enumerate(all_pairs):
                all_indices = set(np.where(pair_indices == num)[0])
                gene_pair = "+".join([ptb_genes[unique_pairs[0]], ptb_genes[unique_pairs[1]]])
                print(" ")
                print("Evaluating for intervention pair {}/{}: {}".format(num+1, len(all_pairs), gene_pair))

                c1 = torch.zeros_like(c_shape).to(device)
                c1[:, unique_pairs[0]] = 1
                c2 = torch.zeros_like(c_shape).to(device)
                c2[:, unique_pairs[1]] = 1

                # Iterate through dataloader to find desired indices
                for i, X in enumerate(dataloader):
                    if i in all_indices:
                        x, y = X[0].to(device), X[1]

                        if len(c1) > len(x):
                            c1 = c1[:len(x), :]
                            c2 = c2[:len(x), :] 

                        with torch.no_grad():
                            y_hat, x_recon, z_mu, z_var, G, _ = model(
                                x, c1, c2, num_interv=numint, temp=temp)

                        gt_x_list.append(x.cpu())
                        pred_x_list.append(x_recon.cpu())

                        gt_y_list.append(y)
                        pred_y_list.append(y_hat.cpu())

                        c_y_list.append(c_shape.cpu())
                        mu_list.append(z_mu.cpu())
                        var_list.append(z_var.cpu())

                    # Limit stacked tensors while iterating through desired indices
                    if len(gt_x_list) >= batch_size:
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

                        # Reset lists
                        pred_x_list, gt_x_list = [], []
                        gt_y_list, pred_y_list = [], []
                        c_y_list, mu_list, var_list = [], [], []

                # Once iterated through all indices, check rest of stacked tensors
                if len(gt_x_list) > 0:
                    # Stack tensors
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

                # After each unique pair, print and save metrics
                print("MSE prediction vs true: {:.4f}".format(np.mean(MSE_pred_true_temp)))
                print("MSE control vs true: {:.4f}".format(np.mean(MSE_ctrl_true_temp)))
                print("MMD prediction vs true: {:.4f}".format(np.mean(MMD_pred_true_temp)))
                print("MMD control vs true: {:.4f}".format(np.mean(MMD_ctrl_true_temp)))
                print("KLD prediction vs true: {:.4f}".format(np.mean(KLD_pred_true_temp)))
                print("KLD control vs true: {:.4f}".format(np.mean(KLD_ctrl_true_temp)))

                # Save MSE
                MSE_pred_true_l.append(np.mean(MSE_pred_true_temp))
                MSE_ctrl_true_l.append(np.mean(MSE_ctrl_true_temp))

                # Save MMD
                MMD_pred_true_l.append(np.mean(MMD_pred_true_temp))
                MMD_ctrl_true_l.append(np.mean(MMD_ctrl_true_temp))

                # Save KLD
                KLD_pred_true_l.append(np.mean(KLD_pred_true_temp))
                KLD_ctrl_true_l.append(np.mean(KLD_ctrl_true_temp))

                # Save results to file
                print(f"{gene_pair},{np.mean(MMD_ctrl_true_temp)},{np.mean(MMD_pred_true_temp)},{np.mean(MSE_ctrl_true_temp)},{np.mean(MSE_pred_true_temp)},{np.mean(KLD_ctrl_true_temp)},{np.mean(KLD_pred_true_temp)}",
                    file=f,
                )

                # Reset lists
                pred_x_list, gt_x_list = [], []
                gt_y_list, pred_y_list = [], []
                c_y_list, mu_list, var_list = [], [], []
                MSE_pred_true_temp, KLD_pred_true_temp, MMD_pred_true_temp = [], [], []
                MSE_ctrl_true_temp, KLD_ctrl_true_temp, MMD_ctrl_true_temp = [], [], []