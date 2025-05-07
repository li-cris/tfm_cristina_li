import json
import os
import sys
import time
import copy
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import numpy as np
import matplotlib

# Torch
import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torch_geometric.loader import DataLoader

# Remember to install gears
# cellgeats module functions
from gears import PertData
#from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
#from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")
path_sypp_scgpt = "../sypp/src/scgpt"
sys.path.insert(0, path_sypp_scgpt)

# Remember to install scgpt
# scGPT module functions
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

# own scGPT functions
from scgpt_utils import model_config_loading, train, eval_perturb, load_pretrained


# Should run it from cris_test folder
# Parallel is sypp/src/scgpt/scripts

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

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

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True

# Settings for loading pretrained model
pretrained_model = True
loaded_model_path = f"{MODEL_DIR_PATH}/scGPT_human"

load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]
# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 8 # NEED LOWER BATCH SIZE
eval_batch_size = 4
epochs = 15 # REMEMBER TO CHANGE THIS
schedule_interval = 1
early_stop = 10

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

# Device cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory where retrained model is saved
save_dir = os.path.join(MODEL_DIR_PATH, f"scgpt_{dataset_name}_{split}_seed_{42}")
os.makedirs(save_dir, exist_ok=True)
print(f"saving to {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, os.path.join(save_dir, "run.log"))
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Loading data
pert_data = PertData(DATA_DIR_PATH)
pert_data.load(data_path=os.path.join(DATA_DIR_PATH, dataset_name))
pert_data.prepare_split(split=split, seed=seed)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

# Load pretrained model configurations or create new ones
pretrained_model = True
loaded_model_configs = model_config_loading(pretrained_model, loaded_model_path, pert_data, special_tokens, logger)

# Set model configurations from configuration file if available
if loaded_model_configs["load_from_config_file"]:
    model_configs = loaded_model_configs["model_configs"]
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

vocab = loaded_model_configs["vocab"]
gene_ids = loaded_model_configs["gene_ids"]
genes = loaded_model_configs["genes"]
n_genes = loaded_model_configs["n_genes"]

# Create and train scGPT
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

# Review, write as function elsewhere when you have time
# Loading complete model
if load_param_prefixs is not None and pretrained_model is not None:
    # only load params that start with the prefix
    model = load_pretrained(model, torch.load(scgpt_model_file), logger=logger)
# When no load_param_prefixs are provided
elif pretrained_model is not None:
    try:
        model.load_state_dict(torch.load(scgpt_model_file))
        logger.info(f"Loading all model params from {scgpt_model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(scgpt_model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

# Clear memory
model.to(device)

# Training settings
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=amp)

# Training loop
best_val_loss = float("inf")
best_val_corr = 0
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Optional checking of allocated memory
    # Total memory and allocated memory on the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory  # Total memory of the GPU
    allocated_memory = torch.cuda.memory_allocated(0)  # Memory currently allocated
    reserved_memory = torch.cuda.memory_reserved(0)  # Memory reserved by PyTorch

    # Free memory (total memory - allocated memory)
    free_memory = total_memory - allocated_memory

    print("Memory stats before training...")
    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free memory: {free_memory / 1e9:.2f} GB")

    train(
        model=model,
        train_loader=train_loader,
        device=device,
        n_genes=n_genes,
        include_zero_gene=include_zero_gene,
        max_seq_len=max_seq_len,
        scaler=scaler,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        CLS=CLS,
        CCE=CCE,
        MVC=MVC,
        ECS=ECS,
        log_interval=log_interval,
        logger=logger,
        epoch=epoch,
        gene_ids=gene_ids,
        amp=amp
    )

    logger.info(f"Epoch {epoch} training finished")

    # Clearing unused memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = torch.cuda.get_device_properties(0).total_memory  # Total memory of the GPU
    allocated_memory = torch.cuda.memory_allocated(0)  # Memory currently allocated
    reserved_memory = torch.cuda.memory_reserved(0)  # Memory reserved by PyTorch

    # Free memory (total memory - allocated memory)
    free_memory = total_memory - allocated_memory
    print("Memory stats before evaluation...")
    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free memory: {free_memory / 1e9:.2f} GB")

   # Validation for each epoch 
    val_res = eval_perturb(
        loader=valid_loader,
        model=model,
        device=device,
        include_zero_gene=include_zero_gene,
        gene_ids=gene_ids
        )
    # pert_cat, pred, truth, pred_de, truth_de

    # First metrics for validation
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    logger.info(f"val_metrics at epoch {epoch}: ")
    logger.info(val_metrics)

    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    # Clearing unused memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = torch.cuda.get_device_properties(0).total_memory  # Total memory of the GPU
    allocated_memory = torch.cuda.memory_allocated(0)  # Memory currently allocated
    reserved_memory = torch.cuda.memory_reserved(0)  # Memory reserved by PyTorch

    # Free memory (total memory - allocated memory)
    free_memory = total_memory - allocated_memory
    print("Memory stats after evaluation...")
    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
    print(f"Free memory: {free_memory / 1e9:.2f} GB")

    # Uses pearson to choose best model
    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {val_score:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    # torch.save(
    #     model.state_dict(),
    #     save_dir / f"model_{epoch}.pt",
    # )

    scheduler.step()


# Save model
# savedir: directory where model is saved, different from dir for pretrained model
torch.save(best_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
