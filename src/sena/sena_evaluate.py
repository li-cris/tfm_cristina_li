# Modules
import os
import sys
import argparse
import json
import pickle
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

path = os.path.abspath('../SENA/src/sena_discrepancy_vae/')
path2 = os.path.abspath('../sypp/src/gears/')
print(path)
sys.path.insert(0, path)
sys.path.insert(0, path2)

from utils import MMD_loss
from kld_loss import compute_kld


# Defining paths and some variables
RESULTS_DIR_PATH = '../cris_test/results/'
savedir = '../cris_test/models/sena_norman_1'
mode = 'double'
numint = 2
device = 'cuda:0'

data_file_map = {
    "train": "train_data.pkl",
    "test": "test_data_single_node.pkl",
    "double": "double_data.pkl",
}

# Configuration settings
config_path = os.path.join(savedir, "config.json")
# Model
model_path = os.path.join(savedir, "best_model.pt")
# Data (double)
data_path = os.path.join(savedir, data_file_map[mode])
# Gene perturbation list
ptb_path = os.path.join(savedir, "ptb_targets.pkl")


# Loading required data
if os.path.exists(data_path):
    with open(data_path, "rb") as f:
        dataloader = pickle.load(f)
else:
    raise FileNotFoundError(f"{mode} data file not found at {data_path}")

# Loading model
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Loading perturbation file
if os.path.exists(ptb_path):
    with open(ptb_path, "rb") as f:
        ptb_genes = pickle.load(f)
else:
    raise FileNotFoundError(f"Perturbation file not found at {ptb_path}")

# Load config
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    # If config file does not exist, use default values or raise an error
    config = {}
    print(f"Warning: Config file not found in {savedir}. Using default parameters.")

MMD_sigma = config.get("MMD_sigma", 200.0)
kernel_num = config.get("kernel_num", 10)
matched_IO = config.get("matched_IO", False)
temp = config.get("temp", 1000.0)
seed = config.get("seed", 42)
latdim = config.get("latdim", 105)
model_name = config.get("name", "example")
batch_size = 10

# Moving model and turning evaluation mode
model = model.to(device)
model.eval()

# Preparation of double interventions
# Find perturbation pairs in data
cidx_list = []
# Grouping by latent space
for i, X in enumerate(tqdm(dataloader, desc="Finding intervention pairs")):
    c = X[2].to(device)
    if i == 0:
        c_shape = c

    idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
    idx_pair = idx.cpu()
    cidx_list.append(idx_pair.numpy())

# Finding unique combinations and index equivalent in whole data
all_pairs, pair_indices = np.unique(cidx_list, axis=0, return_inverse=True)

print("Proceeding with evaluation of intervened pairs.")
# Make results file path.
results_file_path = os.path.join(
    RESULTS_DIR_PATH, f"{model_name}_double_metrics.csv"
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
        mmd_loss_func = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

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
            print("MSE prediction vs true: {}".format(np.mean(MSE_pred_true_temp)))
            print("MSE control vs true: {}".format(np.mean(MSE_ctrl_true_temp)))
            print("MMD prediction vs true: {}".format(np.mean(MMD_pred_true_temp)))
            print("MMD control vs true: {}".format(np.mean(MMD_ctrl_true_temp)))
            print("KLD prediction vs true: {}".format(np.mean(KLD_pred_true_temp)))
            print("KLD control vs true: {}".format(np.mean(KLD_ctrl_true_temp)))

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