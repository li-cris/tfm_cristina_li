import torch

from tadvae.models.gp_transformer_encoder import (
    AttentionLayer,
    GeneEmbeddingLayer,
    GenePathwayTransformerEncoder,
    PathwayEmbeddingLayer,
)


class TestGeneEmbeddingLayer:
    def setup_method(self):
        self.batch_size = 10
        self.n_genes = 100
        self.d_embed = 128
        self.seq_len = 50

    def test_forward(self):
        gene_indices = torch.randint(self.n_genes, (self.batch_size, self.seq_len))
        expression_values = torch.rand((self.batch_size, self.seq_len))

        gene_embedding_layer = GeneEmbeddingLayer(self.n_genes, self.d_embed)
        embedded_genes = gene_embedding_layer(gene_indices, expression_values)

        assert embedded_genes.shape == torch.Size(
            [self.batch_size, self.seq_len, self.d_embed]
        )


class TestPathwayEmbeddingLayer:
    def setup_method(self):
        self.batch_size = 10
        self.n_pathways = 20
        self.d_embed = 128
        self.seq_len = 50

    def test_forward(self):
        pathway_indices = torch.randint(
            self.n_pathways, (self.batch_size, self.seq_len)
        )

        pathway_embedding_layer = PathwayEmbeddingLayer(self.n_pathways, self.d_embed)
        embedded_pathways = pathway_embedding_layer(pathway_indices)

        assert embedded_pathways.shape == torch.Size(
            [self.batch_size, self.seq_len, self.d_embed]
        )


class TestAttentionLayer:
    def setup_method(self):
        self.batch_size = 10
        self.n_heads = 4
        self.q_len = 100
        self.k_len = 20
        self.d_embed = 16
        self.d_head = self.d_embed // self.n_heads

        assert self.d_embed % self.n_heads == 0

    def test_scaled_dot_product_attention(self):
        Q = torch.rand((self.batch_size, self.n_heads, self.q_len, self.d_head))  # noqa: N806
        K = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806
        V = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806

        attention_layer = AttentionLayer(self.d_embed, self.n_heads, mask_type=None)
        output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

        assert output.shape == (self.batch_size, self.n_heads, self.q_len, self.d_head)
        assert attn_weights.shape == (
            self.batch_size,
            self.n_heads,
            self.q_len,
            self.k_len,
        )
        assert torch.allclose(
            attn_weights.sum(dim=-1),
            torch.ones(self.batch_size, self.n_heads, self.q_len),
        )

    def test_scaled_dot_product_attention_hard_mask(self):
        Q = torch.rand((self.batch_size, self.n_heads, self.q_len, self.d_head))  # noqa: N806
        K = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806
        V = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806
        W = torch.randint(2, (self.batch_size, self.q_len, self.k_len))  # noqa: N806

        attention_layer = AttentionLayer(self.d_embed, self.n_heads, mask_type="hard")
        output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V, W)

        assert output.shape == (self.batch_size, self.n_heads, self.q_len, self.d_head)
        assert attn_weights.shape == (
            self.batch_size,
            self.n_heads,
            self.q_len,
            self.k_len,
        )
        assert torch.allclose(
            attn_weights.sum(dim=-1),
            torch.ones(self.batch_size, self.n_heads, self.q_len),
        )

    def test_scaled_dot_product_attention_soft_mask(self):
        Q = torch.rand((self.batch_size, self.n_heads, self.q_len, self.d_head))  # noqa: N806
        K = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806
        V = torch.rand((self.batch_size, self.n_heads, self.k_len, self.d_head))  # noqa: N806
        W = torch.rand((self.batch_size, self.q_len, self.k_len))  # noqa: N806

        attention_layer = AttentionLayer(self.d_embed, self.n_heads, mask_type="soft")
        output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V, W)

        assert output.shape == (self.batch_size, self.n_heads, self.q_len, self.d_head)
        assert attn_weights.shape == (
            self.batch_size,
            self.n_heads,
            self.q_len,
            self.k_len,
        )
        assert torch.allclose(
            attn_weights.sum(dim=-1),
            torch.ones(self.batch_size, self.n_heads, self.q_len),
        )

    def test_forward(self):
        Q = torch.rand((self.batch_size, self.q_len, self.d_embed))  # noqa: N806
        K = torch.rand((self.batch_size, self.k_len, self.d_embed))  # noqa: N806

        attention_layer = AttentionLayer(self.d_embed, self.n_heads, mask_type=None)
        output, attn_weights = attention_layer(Q, K)

        assert output.shape == torch.Size([self.batch_size, self.q_len, self.d_embed])
        assert attn_weights.shape == torch.Size(
            [self.batch_size, self.n_heads, self.q_len, self.k_len]
        )


class TestGenePathwayTransformerEncoder:
    def setup_method(self):
        self.n_genes = 100
        self.n_pathways = 20
        self.d_embed = 16
        self.n_heads = 4
        self.n_layers = 2
        self.batch_size = 5
        self.genes_len = 50
        self.pathways_len = 10

    def test_model_parameters(self):
        W_init = torch.randint(2, (self.n_genes, self.n_pathways))  # noqa: N806

        model = GenePathwayTransformerEncoder(
            self.n_genes,
            self.n_pathways,
            self.d_embed,
            self.n_heads,
            n_layers=1,
            mask_type="soft",
            W_init=W_init,
        )

        expected_parameters = {
            "W_soft": torch.Size([self.n_genes, self.n_pathways]),
            "gene_embedding_layer.embedding.weight": torch.Size(
                [self.n_genes, self.d_embed]
            ),
            "pathway_embedding_layer.embedding.weight": torch.Size(
                [self.n_pathways, self.d_embed]
            ),
            "attention_layers.0.query_linear.weight": torch.Size(
                [self.d_embed, self.d_embed]
            ),
            "attention_layers.0.query_linear.bias": torch.Size([self.d_embed]),
            "attention_layers.0.key_linear.weight": torch.Size(
                [self.d_embed, self.d_embed]
            ),
            "attention_layers.0.key_linear.bias": torch.Size([self.d_embed]),
            "attention_layers.0.value_linear.weight": torch.Size(
                [self.d_embed, self.d_embed]
            ),
            "attention_layers.0.value_linear.bias": torch.Size([self.d_embed]),
            "attention_layers.0.out_linear.weight": torch.Size(
                [self.d_embed, self.d_embed]
            ),
            "attention_layers.0.out_linear.bias": torch.Size([self.d_embed]),
            "output_layer.weight": torch.Size([self.n_pathways, self.d_embed]),
            "output_layer.bias": torch.Size([self.n_pathways]),
        }

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in expected_parameters
                assert param.shape == expected_parameters[name]

    def test_forward(self):
        gene_indices = torch.randint(self.n_genes, (self.batch_size, self.genes_len))
        expression_values = torch.rand((self.batch_size, self.genes_len))
        pathway_indices = torch.randint(
            self.n_pathways, (self.batch_size, self.pathways_len)
        )

        model = GenePathwayTransformerEncoder(
            self.n_genes,
            self.n_pathways,
            self.d_embed,
            self.n_heads,
            self.n_layers,
            mask_type=None,
        )

        output = model(gene_indices, expression_values, pathway_indices)

        assert output.shape == torch.Size([self.batch_size, self.n_pathways])

    def test_forward_hard_mask(self):
        gene_indices = torch.randint(self.n_genes, (self.batch_size, self.genes_len))
        expression_values = torch.rand((self.batch_size, self.genes_len))
        pathway_indices = torch.randint(
            self.n_pathways, (self.batch_size, self.n_pathways)
        )
        W_hard = torch.randint(2, (self.batch_size, self.n_genes, self.n_pathways))  # noqa: N806

        model = GenePathwayTransformerEncoder(
            self.n_genes,
            self.n_pathways,
            self.d_embed,
            self.n_heads,
            self.n_layers,
            mask_type="hard",
        )

        output = model(gene_indices, expression_values, pathway_indices, W_hard)

        assert output.shape == torch.Size([self.batch_size, self.n_pathways])

    def test_forward_soft_mask(self):
        gene_indices = torch.randint(self.n_genes, (self.batch_size, self.genes_len))
        expression_values = torch.rand((self.batch_size, self.genes_len))
        pathway_indices = torch.randint(
            self.n_pathways, (self.batch_size, self.n_pathways)
        )
        W_init = torch.randint(2, (self.n_genes, self.n_pathways))  # noqa: N806

        model = GenePathwayTransformerEncoder(
            self.n_genes,
            self.n_pathways,
            self.d_embed,
            self.n_heads,
            self.n_layers,
            mask_type="soft",
            W_init=W_init,
        )

        output = model(gene_indices, expression_values, pathway_indices)

        assert output.shape == torch.Size([self.batch_size, self.n_pathways])
