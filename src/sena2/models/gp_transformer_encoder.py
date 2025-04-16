from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GeneEmbeddingLayer(nn.Module):
    def __init__(self, n_genes: int, d_embed: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_genes, embedding_dim=d_embed)

    def forward(
        self, gene_indices: torch.Tensor, expression_values: torch.Tensor
    ) -> torch.Tensor:
        gene_embeddings = self.embedding(gene_indices)
        modulated_gene_embeddings = gene_embeddings * expression_values.unsqueeze(-1)
        return modulated_gene_embeddings


class PathwayEmbeddingLayer(nn.Module):
    def __init__(self, n_pathways: int, d_embed: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_pathways, embedding_dim=d_embed)

    def forward(self, pathway_indices: torch.Tensor) -> torch.Tensor:
        pathway_embeddings = self.embedding(pathway_indices)
        return pathway_embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self, d_embed: int, n_heads: int, mask_type: str | None = None
    ) -> None:
        """Multi-head self-attention layer.

        Args:
            d_embed: Embedding dimension.
            n_heads: Number of attention heads.
            mask_type: Type of attention mask. Must be one of None, 'static', or
                'dynamic'.
        """
        super().__init__()

        if d_embed % n_heads != 0:
            raise ValueError("d_embed must be divisible by n_heads.")

        if mask_type not in [None, "static", "dynamic"]:
            raise ValueError("mask_type must be one of None, 'static', or 'dynamic'.")

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.mask_type = mask_type

        self.query_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.key_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.value_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.out_linear = nn.Linear(in_features=d_embed, out_features=d_embed)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,  # noqa: N803
        K: torch.Tensor,  # noqa: N803
        V: torch.Tensor,  # noqa: N803
        W: torch.Tensor = None,  # noqa: N803
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        The optional mask tensor W is used to prevent attention to certain positions.
        It must be provided if self.mask_type is 'static' or 'dynamic'.

        Args:
            Q: Query tensor with shape (batch_size, n_heads, q_len, d_head).
            K: Key tensor with shape (batch_size, n_heads, k_len, d_head).
            V: Value tensor with shape (batch_size, n_heads, k_len, d_head).
            W: Mask tensor with shape (batch_size, q_len, k_len).

        Returns:
            output: Output tensor with shape (batch_size, n_heads, q_len, d_head).
            attn_weights: Attention weights tensor with shape
                (batch_size, n_heads, q_len, k_len).
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)

        if self.mask_type is not None:
            W = W.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # noqa: N806
            if self.mask_type == "static":
                scores = scores.masked_fill(W == 0, float("-inf"))
            if self.mask_type == "dynamic":
                M = torch.sigmoid(W)  # noqa: N806
                scores = scores * M

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(
        self,
        Q: torch.Tensor,  # noqa: N803
        K: torch.Tensor,  # noqa: N803
        W: torch.Tensor = None,  # noqa: N803
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head self-attention.

        Args:
            Q: Query tensor with shape (batch_size, q_len, d_embed).
            K: Key tensor with shape (batch_size, k_len, d_embed).
            W: Mask tensor with shape (batch_size, q_len, k_len).

        Returns:
            output: Output tensor with shape (batch_size, q_len, d_embed).
            attn_weights: Attention weights tensor with shape
                (batch_size, n_heads, q_len, k_len).
        """
        batch_size, q_len, _ = Q.shape
        _, k_len, _ = K.shape

        Q_proj = (  # noqa: N806
            self.query_linear(Q)
            .view(batch_size, q_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        K_proj = (  # noqa: N806
            self.key_linear(K)
            .view(batch_size, k_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        V_proj = (  # noqa: N806
            self.value_linear(K)
            .view(batch_size, k_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q_proj, K_proj, V_proj, W
        )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.d_embed)
        )
        output = self.out_linear(attn_output)

        return output, attn_weights


class GenePathwayTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        d_embed: int,
        n_heads: int,
        n_layers: int,
        mask_type: str | None = None,
        W_init: torch.Tensor = None,  # noqa: N803
        sigma: float = 0.01,
    ) -> None:
        """Gene-pathway transformer encoder.

        Args:
            n_genes: Number of genes.
            n_pathways: Number of pathways.
            d_embed: Embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of attention layers.
            mask_type: Type of attention mask. Must be one of None, 'static', or
                'dynamic'.
            W_init: Initial attention mask tensor with shape (genes_len, n_pathways).
            sigma: Standard deviation of the noise added to the initial attention mask.
        """
        super().__init__()

        if mask_type not in [None, "static", "dynamic"]:
            raise ValueError("mask_type must be one of None, 'static', or 'dynamic'.")

        if mask_type is not None and W_init is None:
            raise ValueError("W_init must be provided when mask_type is not None.")

        self.mask_type = mask_type
        self.W = None

        if self.mask_type is not None:
            self.init_mask(W_init, sigma)

        self.gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
        self.pathway_embedding_layer = PathwayEmbeddingLayer(n_pathways, d_embed)
        self.attention_layers = nn.ModuleList(
            [AttentionLayer(d_embed, n_heads, mask_type) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(in_features=d_embed, out_features=n_pathways)

    def forward(
        self,
        gene_indices: torch.Tensor,
        expression_values: torch.Tensor,
        pathway_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            gene_indices: Gene indices tensor with shape (batch_size, genes_len).
            expression_values: Gene expression values tensor with shape
                (batch_size, genes_len).
            pathway_indices: Pathway indices tensor with shape
                (batch_size, n_pathways).

        Returns:
            pathway_scores: Pathway scores tensor with shape (batch_size, n_pathways).
        """
        batch_size, _ = gene_indices.shape

        if self.W is not None:
            W = self.W.unsqueeze(0).expand(batch_size, -1, -1)  # noqa: N806
        else:
            W = None  # noqa: N806

        gene_embeddings = self.gene_embedding_layer(gene_indices, expression_values)
        pathway_embeddings = self.pathway_embedding_layer(pathway_indices)
        for layer in self.attention_layers:
            gene_embeddings, _ = layer(gene_embeddings, pathway_embeddings, W)
        pathway_scores = self.output_layer(gene_embeddings.mean(dim=1))

        return pathway_scores

    def init_mask(self, W_init: torch.Tensor, sigma: float = 0.01) -> None:  # noqa: N803
        """Initialize the attention mask.

        Can also be used to re-initialize the attention mask (e.g., when switching to
        a new dataset with a different number of genes).

        Args:
            W_init: Attention mask tensor with shape (genes_len, n_pathways).
            sigma: Standard deviation of the noise added to the initial attention mask.
        """
        if self.mask_type == "static":
            if hasattr(self, "W"):
                delattr(self, "W")
            self.register_buffer("W", W_init)

        if self.mask_type == "dynamic":
            self.W = nn.Parameter(W_init + sigma * torch.randn(W_init.shape))

    def get_mask(self) -> torch.Tensor:
        """Return the attention mask, detached from the computation graph.

        This is useful to get and temporarily store a (partially) trained mask when
        switching between different datasets with different numbers of genes.

        Returns:
            W: Attention mask tensor with shape (genes_len, n_pathways).
        """
        if self.mask_type is None:
            raise ValueError("mask_type must be 'static' or 'dynamic' to get the mask.")

        return self.W.detach()
