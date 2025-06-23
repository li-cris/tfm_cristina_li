import os
import json
import torch
import numpy as np

from typing import List, Mapping, Optional

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from gears import PertData


def model_config_loading(pretrained_model: bool,
                         model_directory: str,
                         pert_data: PertData,
                         special_tokens: List[str],
                         logger: None) -> None:
    """
    Loads a pretrained model configuration, vocabulary, and gene mapping.

    If `pretrained_model` is True, loads model parameters, vocabulary, and config 
    from the specified directory. It updates `pert_data` with gene index information 
    for downstream compatibility with scGPT.

    If `pretrained_model` is False, only gene names are extracted from the data,
    and a placeholder vocab is assigned (currently not supported, might generate errors).

    Args:
        pretrained_model (bool): Flag indicating whether to load a pretrained model. Currently only True.
        model_directory (str): Path to the directory containing model files 
                               (`args.json`, `best_model.pt`, `vocab.json`).
        pert_data (PertData): Object containing gene expression data (AnnData format).
        special_tokens (List[str]): List of special tokens (e.g., ["<pad>", "<cls>"]) 
                                    to ensure presence in the vocabulary.
        logger: Logger object for progress and status reporting.

    Returns:
        dict: A dictionary containing the following keys:
            - vocab: The loaded or constructed vocabulary object.
            - load_from_config_file (bool): Indicates whether model config was loaded.
            - model_configs (dict or None): Dictionary of model hyperparameters if loaded.
            - n_genes (int): Number of genes in the dataset.
            - genes (List[str]): List of gene names from the dataset.
            - gene_ids (np.ndarray): Vocabulary indices of the dataset genes.
    """

    if pretrained_model:
        # model_dir = Path(model_directory)
        model_dir = model_directory
        model_config_file = os.path.join(model_dir, "args.json")
        model_file = os.path.join(model_dir, "best_model.pt") # trained model
        vocab_file = os.path.join(model_dir, "vocab.json")

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        # Creates new var id_in_vocab for scgpt
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
            ]

        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
            # embsize, nheads, d_hid, nlayers, n_laters_cls
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )

        load_from_config_file = True

    # Only supports pretrained available
    else:
        print("Only fine-tuning from pretrained model is available for now.")
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = None
        #vocab = Vocab(
        #    VocabPybind(genes + special_tokens, None)
        #    )  # bidirectional lookup [gene <-> int]
        load_from_config_file = False
        model_configs = None

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )

    n_genes = len(genes)

    return {"vocab": vocab,
            "load_from_config_file": load_from_config_file,
            "model_configs" : model_configs,
            "n_genes": n_genes,
            "genes": genes,
            "gene_ids": gene_ids
            }


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
    logger = None
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model