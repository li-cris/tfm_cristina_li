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
    def __init__(self, d_embed: int, n_heads: int) -> None:  # noqa: D107
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads."
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
    ) -> Tuple[
        torch.Tensor,  # output: (batch_size, n_heads, q_len, d_head)
        torch.Tensor,  # attn_weights: (batch_size, n_heads, q_len, k_len)
    ]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(  # noqa: D102
        self,
        x1: torch.Tensor,  # (batch_size, x1_len, d_embed)
        x2: torch.Tensor,  # (batch_size, x2_len, d_embed)
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
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # Reshape and apply output linear layer.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, x1_len, self.d_embed)
        output = self.out_linear(attn_output)

        return output, attn_weights


class AttentionLayerHardMask(AttentionLayer):  # noqa: D101
    def __init__(self, d_embed: int, n_heads: int, mask: torch.Tensor) -> None:  # noqa: D107
        super().__init__(d_embed, n_heads)
        self.mask = mask.unsqueeze(0).unsqueeze(0).expand(-1, n_heads, -1, -1)

    def scaled_dot_product_attention(  # noqa: D102
        self,
        Q: torch.Tensor,  # (batch_size, n_heads, q_len, d_head)  # noqa: N803
        K: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
        V: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
    ) -> Tuple[
        torch.Tensor,  # output: (batch_size, n_heads, q_len, d_head)
        torch.Tensor,  # attn_weights: (batch_size, n_heads, q_len, k_len)
    ]:
        batch_size = Q.shape[0]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)

        # Apply hard mask.
        mask = self.mask.expand(batch_size, -1, -1, -1)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class AttentionLayerSoftMask(AttentionLayer):  # noqa: D101
    def __init__(self, d_embed: int, n_heads: int, mask: torch.Tensor, epsilon: float = 0.01) -> None:  # noqa: D107
        super().__init__(d_embed, n_heads)
        self.W_mask = nn.Parameter(mask + torch.randn(mask.shape) * epsilon)

    def scaled_dot_product_attention(  # noqa: D102
        self,
        Q: torch.Tensor,  # (batch_size, n_heads, q_len, d_head)  # noqa: N803
        K: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
        V: torch.Tensor,  # (batch_size, n_heads, k_len, d_head)  # noqa: N803
    ) -> Tuple[
        torch.Tensor,  # output: (batch_size, n_heads, q_len, d_head)
        torch.Tensor,  # attn_weights: (batch_size, n_heads, q_len, k_len)
    ]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head**0.5)

        # Apply soft mask.
        M_soft = torch.sigmoid(self.W_mask)  # noqa: N806
        M_soft = M_soft.unsqueeze(0).unsqueeze(0)  # noqa: N806
        scores = scores * M_soft  # Element-wise multiplication.

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class GenePathwayTransformerEncoder(nn.Module):  # noqa: D101
    def __init__(self, n_genes: int, n_pathways: int, d_embed: int, n_heads: int, n_layers: int):  # noqa: D107
        super().__init__()

        self.n_pathways = n_pathways
        self.pathway_indices = torch.arange(n_pathways)

        self.gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
        self.pathway_embeddings = PathwayEmbeddingLayer(n_pathways, d_embed)
        self.attention_layers = nn.ModuleList([AttentionLayer(d_embed, n_heads) for _ in range(n_layers)])
        self.output_layer = nn.Linear(in_features=d_embed, out_features=n_pathways)

    def forward(self, gene_indices: torch.Tensor, expression_values: torch.Tensor):  # noqa: D102
        batch_size, _ = gene_indices.shape
        x = self.gene_embedding_layer(gene_indices, expression_values)
        pathway_embeddings = self.pathway_embeddings(self.pathway_indices.unsqueeze(0).expand(batch_size, -1))
        for layer in self.attention_layers:
            x, _ = layer(x, pathway_embeddings)
        pathway_scores = self.output_layer(x.mean(dim=1))
        return pathway_scores


class GenePathwayTransformerEncoderHardMask(GenePathwayTransformerEncoder):  # noqa: D101
    def __init__(self, n_genes: int, n_pathways: int, d_embed: int, n_heads: int, n_layers: int, mask: torch.Tensor):  # noqa: D107
        super().__init__(n_genes, n_pathways, d_embed, n_heads, n_layers)
        self.attention_layers = nn.ModuleList([AttentionLayerHardMask(d_embed, n_heads, mask) for _ in range(n_layers)])


class GenePathwayTransformerEncoderSoftMask(GenePathwayTransformerEncoder):  # noqa: D101
    def __init__(self, n_genes: int, n_pathways: int, d_embed: int, n_heads: int, n_layers: int, mask: torch.Tensor):  # noqa: D107
        super().__init__(n_genes, n_pathways, d_embed, n_heads, n_layers)
        self.attention_layers = nn.ModuleList([AttentionLayerSoftMask(d_embed, n_heads, mask) for _ in range(n_layers)])
