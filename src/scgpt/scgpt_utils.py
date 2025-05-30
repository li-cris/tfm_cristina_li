import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
#from torchtext._torchtext import (
#    Vocab as VocabPybind,
#)

from torch_geometric.loader import DataLoader

import json
import os
from typing import Dict, List, Mapping, Optional, Tuple, Union

from scgpt.model import TransformerGenerator
# from scgpt.loss import (
#     masked_mse_loss,
#     criterion_neg_log_bernoulli,
#     masked_relative_error
# )
#from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

import warnings

from typing import Iterable, List, Tuple, Dict, Union, Optional
import numpy as np
import time
#from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from gears import PertData


def model_config_loading(pretrained_model: bool,
                  model_directory: str,
                  pert_data: PertData,
                  special_tokens: List[str],
                  logger: None) -> None:
    """Load pretrained model if set to true, based on the model_file path."""
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


def train(model: TransformerGenerator, train_loader: torch.utils.data.DataLoader, device: torch.device,
          n_genes: int,
          include_zero_gene: str,
          max_seq_len: int,
          scaler: torch.cuda.amp.GradScaler,
          optimizer: torch.optim.Optimizer,
          criterion,
          scheduler,
          CLS,
          CCE,
          MVC,
          ECS,
          log_interval: int,
          logger,
          epoch: int,
          gene_ids: np.ndarray = None,
          amp: bool = True) -> None:
    """
    Train the model for one epoch.
    """
    print("Start training process...")
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()




def eval_perturb(
    loader: DataLoader,
    model: TransformerGenerator,
    device: torch.device,
    include_zero_gene: str = "all",
    gene_ids: np.ndarray = None
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    # model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    # logvar = []

    print("Starting evaluation...")

    for itr, batch in enumerate(loader):
        if itr%100 == 0:
            print(f"Evaluating batch {itr} / {len(loader)}")

        batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results



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
# TODO: add pert_flags in loader (genes that are perturbed in the list of perturbations)
    print(len(gene_list))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            print(pert)
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for i, batch_data in enumerate(loader):
                batch_data.to(device)
                print(batch_data.x.shape)
                if batch_data.x.dim() == 1:
                    batch_data.x = batch_data.x.unsqueeze(1)
                    pert_flags = [1 if p in gene_list else 0 for p in pert]
                    pert_flags = torch.tensor(pert_flags, dtype=torch.long)
                    pert_flags = pert_flags.repeat(eval_batch_size)
                    pert_flags.to(device)

                    batch_data.x = torch.cat([batch_data.x, pert_flags], dim=1)

                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            # results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
            # results_pred["_".join(pert)] = preds.detach().cpu().numpy()
            results_pred = preds.detach().cpu().numpy()
    return results_pred