import torch

from torch_geometric.loader import DataLoader

from typing import Dict, List, Optional

from scgpt_tool.model import TransformerGenerator

from scgpt.utils import map_raw_id_to_vocab_id, compute_perturbation_metrics

import warnings

import numpy as np
import time

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
    Run model in inference mode using a given data loader. 
    """

    model.eval()
    # model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

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

def validate_perturbation_model(opts, gene_ids, model, pert_data, valid_loader, device):
    """
    Evaluates the perturbation model on a validation set.

    This function runs inference on the validation data and computes relevant
    evaluation metrics to monitor model performance during training.

    Args:
        opts: Options object containing model and training configurations.
        gene_ids (np.ndarray): Array of gene token IDs used for indexing predictions.
        model (nn.Module): The perturbation prediction model (e.g., TransformerGenerator).
        pert_data (PertData): Data object containing the full dataset, including metadata.
        valid_loader (DataLoader): PyTorch DataLoader for the validation subset.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        dict: A dictionary of validation metrics, including performance statistics 
              such as Pearson correlation and mean squared error.
    """

    val_res = eval_perturb(
        loader=valid_loader,
        model=model,
        device=device,
        include_zero_gene=opts.include_zero_gene,
        gene_ids=gene_ids
        )
    
        # First metrics for validation
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )

    return val_metrics