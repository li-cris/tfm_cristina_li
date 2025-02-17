from typing import Tuple  # noqa: D100

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GeneEmbeddingLayer(nn.Module):  # noqa: D101
    def __init__(self, n_genes: int, d_embed: int) -> None:  # noqa: D107
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_genes, embedding_dim=d_embed)

    def forward(self, gene_indices: torch.Tensor, expression_values: torch.Tensor) -> torch.Tensor:  # noqa: D102
        gene_embeddings = self.embedding(gene_indices)
        modulated_gene_embeddings = gene_embeddings * expression_values.unsqueeze(-1)
        return modulated_gene_embeddings


class PathwayEmbeddingLayer(nn.Module):  # noqa: D101
    def __init__(self, n_pathways: int, d_embed: int) -> None:  # noqa: D107
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_pathways, embedding_dim=d_embed)

    def forward(self, pathway_indices: torch.Tensor) -> torch.Tensor:  # noqa: D102
        pathway_embeddings = self.embedding(pathway_indices)
        return pathway_embeddings


class AttentionLayer(nn.Module):  # noqa: D101
    def __init__(self, d_embed: int, n_heads: int, mask_type: str) -> None:  # noqa: D107
        super().__init__()

        if mask_type not in [None, "hard", "soft"]:
            raise ValueError("mask_type must be one of None, 'hard', or 'soft'.")

        self.mask_type = mask_type

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.query_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.key_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.value_linear = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.out_linear = nn.Linear(in_features=d_embed, out_features=d_embed)

    def scaled_dot_product_attention(  # noqa: D102
        self,
        Q: torch.Tensor,  # (batch_size, n_heads, q_len, d_head)  # noqa: N803
        K: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
        V: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
        W: torch.Tensor = None,  # (batch_size, q_len, k_len)  # noqa: N803
    ) -> Tuple[
        torch.Tensor,  # output: (batch_size, n_heads, q_len, d_head)
        torch.Tensor,  # attn_weights: (batch_size, n_heads, q_len, k_len)
    ]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)

        if self.mask_type == "hard":
            W = W.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # noqa: N806
            scores = scores.masked_fill(W == 0, float("-inf"))
        elif self.mask_type == "soft":
            W = W.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # noqa: N806
            M = torch.sigmoid(W)  # noqa: N806
            scores = scores * M  # Element-wise multiplication.

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(  # noqa: D102
        self,
        x1: torch.Tensor,  # (batch_size, x1_len, d_embed)
        x2: torch.Tensor,  # (batch_size, x2_len, d_embed)
        W: torch.Tensor = None,  # (batch_size, x1_len, x2_len)  # noqa: N803
    ) -> Tuple[
        torch.Tensor,  # output: (batch_size, x1_len, d_embed)
        torch.Tensor,  # attn_weights: (batch_size, n_heads, x1_len, x2_len)
    ]:
        batch_size, x1_len, _ = x1.shape
        _, x2_len, _ = x2.shape

        # Compute Q from x1 embeddings, and K and V from x2 embeddings.
        Q = self.query_linear(x1).view(batch_size, x1_len, self.n_heads, self.d_head).transpose(1, 2)  # noqa: N806
        K = self.key_linear(x2).view(batch_size, x2_len, self.n_heads, self.d_head).transpose(1, 2)  # noqa: N806
        V = self.value_linear(x2).view(batch_size, x2_len, self.n_heads, self.d_head).transpose(1, 2)  # noqa: N806

        # Compute attention with Q (from x1) and K/V (from x2).
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, W)

        # Reshape and apply output linear layer.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, x1_len, self.d_embed)
        output = self.out_linear(attn_output)

        return output, attn_weights


class GenePathwayTransformerEncoder(nn.Module):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        n_genes: int,
        n_pathways: int,
        d_embed: int,
        n_heads: int,
        n_layers: int,
        mask_type: str,
        W_init: torch.Tensor = None,  # (n_genes, n_pathways)  # noqa: N803
    ):
        super().__init__()

        if mask_type not in [None, "hard", "soft"]:
            raise ValueError("mask_type must be one of None, 'hard', or 'soft'.")

        if mask_type == "soft" and W_init is None:
            raise ValueError("W_init must be provided when mask_type is 'soft'.")

        self.mask_type = mask_type

        if mask_type == "soft":
            epsilon = 0.01
            self.W_soft = nn.Parameter(W_init + torch.randn(W_init.shape) * epsilon)

        self.gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
        self.pathway_embeddings = PathwayEmbeddingLayer(n_pathways, d_embed)
        self.attention_layers = nn.ModuleList([AttentionLayer(d_embed, n_heads, mask_type) for _ in range(n_layers)])
        self.output_layer = nn.Linear(in_features=d_embed, out_features=n_pathways)

    def forward(  # noqa: D102
        self,
        gene_indices: torch.Tensor,
        expression_values: torch.Tensor,
        pathway_indices: torch.Tensor,
        W_hard: torch.Tensor = None,  # noqa: N803
    ) -> torch.Tensor:
        batch_size = gene_indices.shape[0]

        if self.mask_type == "hard":
            W = W_hard[:, gene_indices[1], :]  # noqa: N806
        elif self.mask_type == "soft":
            W = self.W_soft[gene_indices[1], :].unsqueeze(0).expand(batch_size, -1, -1)  # noqa: N806
        else:
            W = None  # noqa: N806

        gene_embeddings = self.gene_embedding_layer(gene_indices, expression_values)
        pathway_embeddings = self.pathway_embeddings(pathway_indices)
        for layer in self.attention_layers:
            gene_embeddings, _ = layer(gene_embeddings, pathway_embeddings, W)
        pathway_scores = self.output_layer(gene_embeddings.mean(dim=1))
        return pathway_scores
