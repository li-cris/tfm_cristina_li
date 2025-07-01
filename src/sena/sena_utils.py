import os
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm


def check_and_load_paths(data_path: str, model_path: str, ptb_path: str, config_path: str, mode: str, savedir: str) -> None:
    """Check if the given path exists and is a directory."""
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
    return(dataloader, model, ptb_genes, config)




def find_pert_pairs(dataloader: str, device: str) -> None:
    """Find perturbation pairs from the dataloader."""
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

    return all_pairs, pair_indices, c_shape