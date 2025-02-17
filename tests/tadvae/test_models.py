import torch  # noqa: D100

from tadvae.models import (
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
    gene_indices = torch.randint(0, n_genes, (batch_size, seq_len))
    expression_values = torch.rand((batch_size, seq_len))

    # Initialize gene embedding layer and compute gene embeddings.
    gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
    embedded_genes = gene_embedding_layer(gene_indices, expression_values)

    # Check the shape of the output.
    assert embedded_genes.shape == torch.Size([batch_size, seq_len, d_embed])


def test_pathway_embedding_layer():  # noqa: D103
    batch_size = 2
    n_pathways = 20
    d_embed = 128
    seq_len = 50

    # Simulate input.
    pathway_indices = torch.randint(0, n_pathways, (batch_size, seq_len))

    # Initialize pathway embedding layer and compute pathway embeddings.
    pathway_embedding_layer = PathwayEmbeddingLayer(n_pathways, d_embed)
    embedded_pathways = pathway_embedding_layer(pathway_indices)

    # Check the shape of the output.
    assert embedded_pathways.shape == torch.Size([batch_size, seq_len, d_embed])


def test_attention_layer_scaled_dot_product_attention():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    d_head = d_embed // n_heads

    # Simulate input.
    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806

    # Initialize attention layer and compute scaled dot-product attention.
    attention_layer = AttentionLayer(d_embed, n_heads, mask_type=None)
    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

    # Check the shape of the output.
    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, q_len), atol=1e-3)


def test_attention_layer_scaled_dot_product_attention_hard_mask():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    d_head = d_embed // n_heads

    # Simulate input.
    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    W = torch.randint(0, 2, (batch_size, q_len, k_len))  # noqa: N806

    # Initialize attention layer and compute scaled dot-product attention.
    attention_layer = AttentionLayer(d_embed, n_heads, mask_type="hard")
    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V, W)

    # Check the shape of the output.
    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, q_len), atol=1e-3)


def test_attention_layer_scaled_dot_product_attention_soft_mask():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    d_head = d_embed // n_heads

    # Simulate input.
    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    W = torch.randint(0, 2, (batch_size, q_len, k_len)).float()  # noqa: N806

    # Initialize attention layer and compute scaled dot-product attention.
    attention_layer = AttentionLayer(d_embed, n_heads, mask_type="soft")
    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V, W)

    # Check the shape of the output.
    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, q_len), atol=1e-3)


def test_attention_layer():  # noqa: D103
    batch_size = 2
    n_heads = 4
    x1_len = 10
    x2_len = 15
    d_embed = 16

    attention_layer = AttentionLayer(d_embed, n_heads, mask_type=None)

    x1 = torch.rand(batch_size, x1_len, d_embed)
    x2 = torch.rand(batch_size, x2_len, d_embed)

    output, attn_weights = attention_layer(x1, x2)

    assert output.shape == torch.Size([batch_size, x1_len, d_embed])
    assert attn_weights.shape == torch.Size([batch_size, n_heads, x1_len, x2_len])


def test_gene_pathway_transformer_encoder():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 100
    d_embed = 16
    n_heads = 4
    n_layers = 2
    mask_type = None
    batch_size = 5
    seq_len = 20

    # Instantiate the model.
    model = GenePathwayTransformerEncoder(n_genes, n_pathways, d_embed, n_heads, n_layers, mask_type)

    # Generate random test inputs.
    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
    expression_values = torch.rand(batch_size, seq_len)
    pathway_indices = torch.randint(low=0, high=n_pathways, size=(batch_size, seq_len))

    # Min-max normalization for the expression values.
    min_expression_value = torch.min(expression_values)
    max_expression_value = torch.max(expression_values)
    normalized_expression_values = (expression_values - min_expression_value) / (
        max_expression_value - min_expression_value
    )

    # Run a forward pass.
    output = model(gene_indices, normalized_expression_values, pathway_indices)

    assert output.shape == torch.Size([batch_size, n_pathways])


def test_gene_pathway_transformer_encoder_hard_mask():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 101
    d_embed = 16
    n_heads = 4
    n_layers = 2
    mask_type = "hard"
    batch_size = 5
    seq_len = 20

    # Instantiate the model.
    model = GenePathwayTransformerEncoder(n_genes, n_pathways, d_embed, n_heads, n_layers, mask_type)

    # Generate random test inputs.
    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
    expression_values = torch.rand(batch_size, seq_len)
    pathway_indices = torch.randint(low=0, high=n_pathways, size=(batch_size, n_pathways))
    W_hard = torch.randint(0, 2, (batch_size, n_genes, n_pathways))  # noqa: N806

    # Min-max normalization for the expression values.
    min_expression_value = torch.min(expression_values)
    max_expression_value = torch.max(expression_values)
    normalized_expression_values = (expression_values - min_expression_value) / (
        max_expression_value - min_expression_value
    )

    # Run a forward pass.
    output = model(gene_indices, normalized_expression_values, pathway_indices, W_hard)

    assert output.shape == torch.Size([batch_size, n_pathways])


def test_gene_pathway_transformer_encoder_soft_mask():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 101
    d_embed = 16
    n_heads = 4
    n_layers = 2
    mask_type = "soft"
    batch_size = 5
    seq_len = 20

    # Generate random test inputs.
    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
    expression_values = torch.rand(batch_size, seq_len)
    pathway_indices = torch.randint(low=0, high=n_pathways, size=(batch_size, n_pathways))
    W_init = torch.randint(0, 2, (n_genes, n_pathways))  # noqa: N806

    # Instantiate the model.
    model = GenePathwayTransformerEncoder(n_genes, n_pathways, d_embed, n_heads, n_layers, mask_type, W_init)

    # Min-max normalization for the expression values.
    min_expression_value = torch.min(expression_values)
    max_expression_value = torch.max(expression_values)
    normalized_expression_values = (expression_values - min_expression_value) / (
        max_expression_value - min_expression_value
    )

    # Run a forward pass.
    output = model(gene_indices, normalized_expression_values, pathway_indices)

    assert output.shape == torch.Size([batch_size, n_pathways])
