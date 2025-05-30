import os
import sys
import time
import numpy as np
import pandas as pd

# Torch
import torch

# GEARS
from gears import PertData

# Remember to install scgpt
# scGPT module functions
import scgpt as scg
from scgpt.model import TransformerGenerator

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

sys.path.insert(0, "../")
path_sypp_scgpt = "../sypp/src/scgpt"
sys.path.insert(0, path_sypp_scgpt)
sys.path.insert(0, os.path.join("../sypp/src"))

# own scGPT and metric functions
from scgpt_utils import load_pretrained, predict
from gears_tools.kld_loss import compute_kld
from gears_tools.mmd_loss import MMDLoss

PREDICT_DOUBLE = True
MODEL_DIR_PATH = './models'
RESULT_DIR_PATH = './results'
DATA_DIR_PATH = './data'


# Somo parameters and settings
seed = 42
set_seed(seed)

# dataset and evaluation choices
dataset_name = "norman_alt" # norman_alt, others...
split = "simulation"

# settings for data preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536
eval_batch_size = 4

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path settings
loaded_model_name = f"scgpt_{dataset_name}_{split}_seed_{seed}"
loaded_model_path = f"{MODEL_DIR_PATH}/{loaded_model_name}"

result_savedir = os.path.join(RESULT_DIR_PATH, "scgpt")
os.makedirs(result_savedir, exist_ok=True)
print(f"saving to {result_savedir}")

# Logger
logger = scg.logger
scg.utils.add_file_handler(logger, os.path.join(result_savedir, "predict_evaluate.log"))
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load GEARS dataset
pert_data = PertData(DATA_DIR_PATH)
pert_data.load(data_path=os.path.join(DATA_DIR_PATH, dataset_name))
pert_data.prepare_split(split=split, seed=seed)
# pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

# Get list of perturbations to predict
if PREDICT_DOUBLE:
    # Get all double perturbations.
    double_perturbations = set(
        [c for c in pert_data.adata.obs["condition"] if "ctrl" not in c]
    )
    print(f"Number of double perturbations: {len(double_perturbations)}")


# Load model
# Get gene_ids from the pretrained foundation model
foundation_model_path = f"{MODEL_DIR_PATH}/scGPT_human"

# Load vocab for gene_ids
vocab_file = os.path.join(foundation_model_path, "vocab.json")
vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
       if s not in vocab:
           vocab.append_token(s)
vocab.set_default_index(vocab["<pad>"])
genes = pert_data.adata.var["gene_name"].tolist()
gene_ids = np.array(
     [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
     )

# settings for the model (Review this)
embsize = 256  # embedding dimension
d_hid = 256  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer
ntokens = len(vocab)  # size of vocabulary

# Load Transformer model based on configurations
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    use_fast_transformer=use_fast_transformer,
)

scgpt_model_file = os.path.join(loaded_model_path, "best_model.pt")
model = load_pretrained(model, torch.load(scgpt_model_file), logger=logger)
model.to(device)

# Predict doubles
# List of doubles

# Random set of ctrl samples
pool_size=20
ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
np.random.seed(seed)
random_indices = np.random.choice(
      ctrl_adata.n_obs, size=pool_size, replace=False
      )
ctrl_geps = ctrl_adata[random_indices, :]
ctrl_geps_tensor = torch.tensor(ctrl_geps.X.toarray())


# Setting up csv file for metrics
double_results_file_path = os.path.join(
    result_savedir, f"{loaded_model_name}_double_metrics.csv")

with open(file=double_results_file_path, mode="w") as f:
    print(
         "double,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred,kld_true_vs_ctrl,kld_true_vs_pred",
         file=f,
         )

    mean_result_pred = {}
    for i, double in enumerate(double_perturbations):

        print(f"Evaluating double {double}: {i + 1}/{len(double_perturbations)}")
        pert_list = [double.split("+")] # This should be a list, but it's only one double pert

        # Perturbation prediction for pool_size for current double
        result_pred = predict(model=model, pert_list=pert_list, pert_data=pert_data,
                            eval_batch_size=eval_batch_size, include_zero_gene=include_zero_gene, gene_ids=gene_ids, pool_size=pool_size)

        mean_result_pred["_".join(pert_list)] = np.mean(result_pred, axis=0)

        # Getting some samples for the current double pert
        np.random.seed(seed)
        double_adata = pert_data.adata[pert_data.adata.obs["condition"] == double]
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

        if i > 1:
             print("Stopping after 2 doubles for testing purposes.")
             break

print(f"Metrics saved to {double_results_file_path}.")
# Saving predictions as another csv
predictions_df = pd.DataFrame.from_dict(mean_result_pred, orient='index')
predictions_df.columns = pert_data.adata.var_names
# Turn index into column
predictions_df.reset_index(inplace=True, names='double')

# Save predictions
prediction_file_path = os.path.join(result_savedir, f"{loaded_model_name}_double.csv")
predictions_df.to_csv(prediction_file_path, index=False)
print(f"Predictions saved to {prediction_file_path}.")