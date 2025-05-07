import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import torch.nn as nn
import scanpy as sc

path = os.path.abspath('../sypp/src/')
print(path)
sys.path.insert(0, path)
sys.path.insert(0, os.path.abspath('../sypp/src/lgem/'))
sys.path.insert(0, os.path.abspath('../'))

from lgem.models import (
    LinearGeneExpressionModelLearned,
    LinearGeneExpressionModelOptimized,
)
from cris_test.single_norman_utils import separate_data, predict_evaluate_lgem_double

# Loading dataloaders as well as model
MODEL_DIR_PATH = '../cris_test/models/'
RESULT_DIR_PATH = '../cris_test/results/'

# Parameters.
seed = 40 # prev 42
num_runs = 1
batch_size = 8
n_epochs = 25000
run_name = f"lgem_seed_{seed}_runs_{num_runs}"

# Control baseline
adata_filepath = '/wdir/tfm/SENA/data/Norman2019_raw.h5ad'
norman = sc.read(adata_filepath)
dataset_name = "norman"
_, _, norman_ctrl = separate_data(adata = norman, dataset_name = dataset_name)

# PyTorch setup.
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = f"lgem_seed_{seed}"
savedir = os.path.join(MODEL_DIR_PATH, model_name)

# Load dataset (dataloader)
test_dataloader = torch.load(os.path.join(savedir, "test_dataloader.pt"))
perturbation_list = torch.load(os.path.join(savedir, "perts.pt"))
perturbation_list = perturbation_list["perts"]

# Load Embeddings
G = torch.load(os.path.join(savedir, "G.pt"))
P = torch.load(os.path.join(savedir, "P.pt"))
Y = torch.load(os.path.join(savedir, "Y.pt"))
b = torch.load(os.path.join(savedir, "b.pt"))

# Optimized model
model_optimized = LinearGeneExpressionModelOptimized(Y.T, G, P, b)
model_optimized.load_state_dict(torch.load(os.path.join(savedir, "optimized_best_model.pt")))

double_perts_list_op, double_predictions_op, ground_truth, loss_op = predict_evaluate_lgem_double(model_optimized, device, test_dataloader, perturbation_list)

# Learned model
model_learned = LinearGeneExpressionModelLearned(G, b)
model_learned.load_state_dict(torch.load(os.path.join(savedir, "learned_best_model.pt")))
_, double_predictions_learn, _, loss_learn= predict_evaluate_lgem_double(model_learned, device, test_dataloader, perturbation_list)

# Randomly chosen control cells for baseline
rand_idx = np.random.randint(low=0, high=norman_ctrl.X.shape[0], size=len(double_perts_list_op))
baseline_control = norman_ctrl.X[rand_idx, :].toarray()

# Array conversion
double_predictions_op = np.asarray(double_predictions_op)
double_predictions_learn = np.asarray(double_predictions_learn)
baseline_control = np.asarray(baseline_control)

# Calculate pred - baseline MSE
mse_op = np.mean((double_predictions_op - baseline_control) ** 2, axis = 1)
mse_learn = np.mean((double_predictions_learn - baseline_control) ** 2, axis = 1)
print("Finalised MSE calculations.")

# Save metrics to result dir
result_df = pd.DataFrame({"double": double_perts_list_op,
                          "mse_pred_vs_control_op": mse_op,
                          "mse_pred_vs_control_learn": mse_learn,
                          "mse_pred_vs_true_op": loss_op,
                          "mse_pred_vs_true_learn": loss_learn})

result_df.to_csv(os.path.join(RESULT_DIR_PATH, f"{model_name}_double_metrics.csv"), index=False)
print(f"Results saved to {os.path.join(RESULT_DIR_PATH, f'{model_name}_double_metrics.csv')}")



