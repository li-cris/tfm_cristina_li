import torch  # noqa: D100

from tadvae.model import (
    AttentionLayer,
    AttentionLayerHardMask,
    AttentionLayerSoftMask,
    GeneEmbeddingLayer,
    GenePathwayTransformerEncoder,
    GenePathwayTransformerEncoderHardMask,
    GenePathwayTransformerEncoderSoftMask,
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

    # Initialize gene embedding layer and compute gene embeddings.
    gene_embedding_layer = GeneEmbeddingLayer(n_genes, d_embed)
    embedded_genes = gene_embedding_layer(gene_indices, expression_values)

    assert embedded_genes.shape == torch.Size([batch_size, seq_len, d_embed])


def test_pathway_embedding_layer():  # noqa: D103
    batch_size = 2
    n_pathways = 20
    d_embed = 128

    # Simulate input.
    pathway_indices = torch.arange(start=0, end=n_pathways).repeat(batch_size, 1)

    # Initialize pathway embedding layer and compute pathway embeddings.
    pathway_embedding_layer = PathwayEmbeddingLayer(n_pathways, d_embed)
    embedded_pathways = pathway_embedding_layer(pathway_indices)

    assert embedded_pathways.shape == torch.Size([batch_size, n_pathways, d_embed])


def test_attention_layer_scaled_dot_product_attention():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    assert d_embed % n_heads == 0
    d_head = d_embed // n_heads

    attention_layer = AttentionLayer(d_embed, n_heads)

    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806

    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, q_len), atol=1e-6)


def test_attention_layer_hard_mask_scaled_dot_product_attention():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    assert d_embed % n_heads == 0
    d_head = d_embed // n_heads
    mask = torch.randint(low=0, high=2, size=(q_len, k_len))

    attention_layer = AttentionLayerHardMask(d_embed, n_heads, mask)

    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806

    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head**0.5)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    expected_attn_weights = torch.softmax(masked_scores, dim=-1)
    expected_output = torch.matmul(expected_attn_weights, V)

    assert torch.allclose(output, expected_output, atol=1e-6)
    assert torch.allclose(attn_weights, expected_attn_weights, atol=1e-6)


def test_attention_layer_soft_mask_scaled_dot_product_attention():  # noqa: D103
    batch_size = 2
    n_heads = 2
    q_len = 5
    k_len = 7
    d_embed = 16
    assert d_embed % n_heads == 0
    d_head = d_embed // n_heads
    mask = torch.randint(low=0, high=2, size=(q_len, k_len)).float()

    attention_layer = AttentionLayerSoftMask(d_embed, n_heads, mask)

    Q = torch.rand(batch_size, n_heads, q_len, d_head)  # noqa: N806
    K = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806
    V = torch.rand(batch_size, n_heads, k_len, d_head)  # noqa: N806

    output, attn_weights = attention_layer.scaled_dot_product_attention(Q, K, V)

    assert output.shape == (batch_size, n_heads, q_len, d_head)
    assert attn_weights.shape == (batch_size, n_heads, q_len, k_len)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head**0.5)
    M_soft = torch.sigmoid(attention_layer.W_mask)  # noqa: N806
    expected_scores = scores * M_soft
    expected_attn_weights = torch.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_attn_weights, V)

    assert torch.allclose(output, expected_output, atol=1e-6)
    assert torch.allclose(attn_weights, expected_attn_weights, atol=1e-6)


def test_attention_layer():  # noqa: D103
    batch_size = 2
    n_heads = 4
    x1_len = 10
    x2_len = 15
    d_embed = 16
    assert d_embed % n_heads == 0

    attention_layer = AttentionLayer(d_embed, n_heads)

    x1 = torch.rand(batch_size, x1_len, d_embed)
    x2 = torch.rand(batch_size, x2_len, d_embed)

    output, attn_weights = attention_layer(x1, x2)

    assert output.shape == torch.Size([batch_size, x1_len, d_embed])
    assert attn_weights.shape == torch.Size([batch_size, n_heads, x1_len, x2_len])


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
    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
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


def test_gene_pathway_transformer_encoder_hard_mask():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 8
    d_embed = 16
    n_heads = 4
    n_layers = 2
    batch_size = 5
    seq_len = 20
    mask = torch.randint(low=0, high=2, size=(seq_len, n_pathways)).float()

    model = GenePathwayTransformerEncoderHardMask(n_genes, n_pathways, d_embed, n_heads, n_layers, mask)

    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
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


def test_gene_pathway_transformer_encoder_soft_mask():  # noqa: D103
    # Define model parameters.
    n_genes = 100
    n_pathways = 8
    d_embed = 16
    n_heads = 4
    n_layers = 2
    batch_size = 5
    seq_len = 20
    mask = torch.randint(low=0, high=2, size=(seq_len, n_pathways)).float()

    model = GenePathwayTransformerEncoderSoftMask(n_genes, n_pathways, d_embed, n_heads, n_layers, mask)

    gene_indices = torch.randint(low=0, high=n_genes, size=(batch_size, seq_len))
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
