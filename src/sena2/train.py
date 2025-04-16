import os
from collections import defaultdict

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from .models.cmvae import CMVAE
from .utils import LossFunction


# Train CMVAE to data
def train(
    dataloader: torch.utils.data.DataLoader,
    opts: "Options",  # noqa: F821 # type: ignore
    device: torch.device,
    savedir: str,
    logger,
    data_handler,
    gpmask: torch.Tensor,
    gene_indices: torch.Tensor,
    pathway_indices: torch.Tensor,
) -> None:
    logger.info(f"Started training on device: {device}")

    # Expand gene and pathway indices to batch size.
    batch_size = opts.batch_size
    gene_indices = gene_indices.unsqueeze(0).expand(batch_size, -1)
    pathway_indices = pathway_indices.unsqueeze(0).expand(batch_size, -1)

    # Move tensors to device.
    gpmask.to(device)
    gene_indices = gene_indices.to(device)
    pathway_indices = pathway_indices.to(device)

    # Initialize model
    cmvae = (
        CMVAE(
            dim=opts.dim,
            z_dim=opts.latdim,
            c_dim=opts.cdim,
            device=device,
            mode=opts.model,
            gos=data_handler.gos,
            rel_dict=data_handler.rel_dict,
            gpmask=gpmask,
            gene_indices=gene_indices,
            pathway_indices=pathway_indices,
            sena_lambda=opts.sena_lambda,
        )
        .double()
        .to(device)
    )

    optimizer = Adam(params=cmvae.parameters(), lr=opts.lr)

    cmvae.train()
    logger.info(f"Training for {opts.epochs} epochs...")

    # Loss parameter schedules
    beta_schedule = torch.cat(
        [torch.zeros(10), torch.linspace(0, opts.mxBeta, opts.epochs - 10)]
    )
    alpha_schedule = torch.cat(
        [
            torch.zeros(5),
            torch.linspace(0, opts.mxAlpha, int(opts.epochs / 2) - 5),
            torch.full((opts.epochs - int(opts.epochs / 2),), opts.mxAlpha),
        ]
    )
    temp_schedule = torch.cat(
        [torch.ones(5), torch.linspace(1, opts.mxTemp, opts.epochs - 5)]
    )

    min_train_loss = np.inf

    # init loss function class
    loss_f = LossFunction(
        MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num, matched_IO=opts.matched_IO
    )

    # Training loop
    for epoch in range(opts.epochs):
        epoch_losses = defaultdict(float)

        # Using tqdm for progress bar during batch iteration
        for batch in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{opts.epochs}", unit="batch"
        ):
            x, y, c = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var, G, bc = cmvae(  # noqa: N806
                x, c, c, num_interv=1, temp=temp_schedule[epoch]
            )
            mmd_loss, recon_loss, kl_loss, l1_loss = loss_f.compute_loss(
                y_hat, y, x_recon, x, z_mu, z_var, G
            )
            loss = (
                alpha_schedule[epoch] * mmd_loss
                + recon_loss
                + beta_schedule[epoch] * kl_loss
                + opts.lmbda * l1_loss
            )
            loss.backward()

            if opts.grad_clip:
                torch.nn.utils.clip_grad_value_(cmvae.parameters(), clip_value=0.5)

            optimizer.step()

            # Log batch losses
            epoch_losses["loss"] += loss.item()
            epoch_losses["mmd_loss"] += mmd_loss.item()
            epoch_losses["recon_loss"] += recon_loss.item()
            epoch_losses["kl_loss"] += kl_loss.item()
            epoch_losses["l1_loss"] += l1_loss.item()

        # Log average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= len(dataloader)

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Loss={epoch_losses['loss']:.6f}, "
            f"MMD={epoch_losses['mmd_loss']:.6f}, "
            f"MSE={epoch_losses['recon_loss']:.6f}, "
            f"KL={epoch_losses['kl_loss']:.6f}, "
            f"L1={epoch_losses['l1_loss']:.6f}"
        )

        # Save the best model
        current_loss = sum(epoch_losses.values()) / len(epoch_losses)
        if current_loss < min_train_loss:
            min_train_loss = current_loss
            torch.save(cmvae, os.path.join(savedir, "best_model.pt"))
            logger.info(f"Best model saved at epoch {epoch + 1}")
