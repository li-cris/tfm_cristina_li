import torch  # noqa: D100

from tadvae.model import (
    AttentionLayer,
    GeneEmbeddingLayer,
    GenePathwayTransformerEncoder,
    PathwayEmbeddingLayer,
)


def test_gene_embedding_layer():  # noqa: D103
    batch_size = 2
    n_genes = 100
    d_embed = 128
    seq_len = 50

    # Simulate input.
    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
    expression_values = torch.rand(batch_size, seq_len)

    # Initialize embedding layer and compute gene embeddings.
    gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
    embedded_genes = gene_embedding_layer(gene_indices, expression_values)

    assert embedded_genes.shape == torch.Size([batch_size, seq_len, d_embed])


def test_pathway_embedding_layer():  # noqa: D103
    batch_size = 2
    n_pathways = 20
    d_embed = 128

    # Simulate input.
    pathway_indices = torch.arange(start=0, end=n_pathways).repeat(batch_size, 1)

    # Initialize embedding layer and compute pathway embeddings.
    pathway_embedding_layer = PathwayEmbeddingLayer(n_pathways, d_embed)
    embedded_pathways = pathway_embedding_layer(pathway_indices)

    assert embedded_pathways.shape == torch.Size([batch_size, n_pathways, d_embed])


def test_scaled_dot_product_attention():  # noqa: D103
    batch_size = 2
    n_heads = 2
    seq_len = 7
    d_embed = 16

    assert d_embed % n_heads == 0

    d_head = d_embed // n_heads

    attention_layer = AttentionLayer(d_embed, n_heads)

    Q = torch.rand(batch_size, n_heads, seq_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, seq_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, seq_len, d_head)  # noqa: N806

    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

    assert output.shape == (batch_size, n_heads, seq_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len), atol=1e-6)


def test_gene_pathway_transformer_encoder():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 8
    d_embed = 16
    n_heads = 4
    n_layers = 2
    batch_size = 5
    seq_len = 20

    # Instantiate the model.
    model = GenePathwayTransformerEncoder(n_genes, n_pathways, d_embed, n_heads, n_layers)

    # Generate random test inputs.
    gene_indices = torch.randint(0, n_genes, (batch_size, seq_len))
    expression_values = torch.rand(batch_size, seq_len)

    # Min-max normalization for the expression values.
    min_expression_value = torch.min(expression_values)
    max_expression_value = torch.max(expression_values)
    normalized_expression_values = (expression_values - min_expression_value) / (
        max_expression_value - min_expression_value
    )

    # Run a forward pass.
    output = model(gene_indices, normalized_expression_values)

    assert output.shape == torch.Size([batch_size, n_pathways])
    print(output)
