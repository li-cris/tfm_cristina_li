import torch
import torch.nn as nn

from .autoencoder import Autoencoder
from .gp_transformer_encoder import GenePathwayTransformerEncoder


class TAAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        d_embed: int,
        n_heads: int,
        n_layers: int,
        d_hidden: int,
        mask_type: str,
        W_init: torch.Tensor = None,  # noqa: N803
    ) -> None:
        super().__init__()

        self.gene_pathway_transformer_encoder = GenePathwayTransformerEncoder(
            n_genes,
            n_pathways,
            d_embed,
            n_heads,
            n_layers,
            mask_type,
            W_init,
        )

        self.autoencoder = Autoencoder(
            input_dim=n_pathways, hidden_dim=d_hidden, output_dim=n_genes
        )

    def forward(
        self,
        gene_indices: torch.Tensor,
        expression_values: torch.Tensor,
        pathway_indices: torch.Tensor,
        W_hard: torch.Tensor = None,  # noqa: N803
    ) -> torch.Tensor:
        pathway_embeddings = self.gene_pathway_transformer_encoder(
            gene_indices, expression_values, pathway_indices, W_hard
        )

        reconstructed_expression_values = self.autoencoder(pathway_embeddings)

        return reconstructed_expression_values
