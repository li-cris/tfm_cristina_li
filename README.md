# Transformer-Based Activity Discrepancy Variational Autoencoder

## Overview

The NetActivity layer maps a gene expression profile to a set of gene set/biological pathway activity scores.
It involves a so-called pathway mask that only allows those connections of genes to pathways that are present in the mask.

We now extend this model such that it can ingest data from multiple scRNA-seq datasets.
Hence, the model needs to be able to handle variable-sized inputs (different numbers of measured genes) and output (different numbers of pathways).

To accomplish this, we use a Transformer-like architecture with attention to model the pathway mask dynamically.
The key idea is to replace the fixed pathway mask with an attention-based mechanism that learns context-dependent relationships between genes and pathways.
This would also allow handling datasets with different genes and pathways flexibly.

The approach involves the following steps:

1. Tokenize the input:
    - We will represent each gene as a token with its expression level as a feature.
    - Each dataset can have a variable number of measured genes, so the input will be a sequence of gene tokens.
2. Learnable embeddings for genes and pathways:
    - Assign each gene a learnable embedding vector.
    - Assign each pathway a learnable embedding vector.
3. Attention-based masking:
    - Use a cross-attention mechanism where genes attend to pathways.
    - The pathway mask is integrated into the attention mechanism using either a hard masking or soft masking approach.
4. Transformer encoder for gene representation:
    - Use a Transformer encoder to learn contextualized representations of genes.

## Masking in Attention Scores

### Approach 1: Hard Masking in Attention Scores

This ensures attention is only computed for valid gene-pathway connections by setting attention scores for invalid pairs to $-\infty$ before applying the softmax.
This will lead to zeros for invalid pairs in the attention pattern.

We define the pathway mask $M \in \mathbb{R}^{n_p, n_g}$, with $n_p$ being the number of pathways and $n_g$ being the number of genes, as:

$$
M_{ij} =
    \begin{cases}
        1, & \text{if } \text{gene } i \text{ is in pathway } i \\
        0, & \text{otherwise}
    \end{cases}
$$

The modified attention formula then applies additive masking to prevent attention outside of the predefined connections:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{logits}} \right) V
$$

where:

$$
M_{\text{logits}} =
    \begin{cases}
        0, & \text{if } M_{ij} = 1 \\
        -\infty, & \text{if } M_{ij} = 0
\end{cases}
$$

This ensures that genes only contribute to pathways where they are valid.
The softmax function ignores masked connections since exponentiating $-\infty$ results in zero.

### Approach 2: Soft Masking via Learnable Pathway Mask

Soft masking allows flexibility by learning the influence of genes on pathways.
Instead of strict binary constraints, we introduce a learnable weight matrix $W_{\text{mask}}$:

$$
M_{\text{soft}} = \sigma(W_{\text{mask}})
$$

where $\sigma(x)$ is the sigmoid function that keeps values in the range $[0,1]$.
The attention formula is:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left(\frac{QK^T}{\sqrt{d_k}} \odot M_{\text{soft}} \right) V
$$


Here, $\odot$ is the element-wise multiplication (Hadamard product).
This means:
If a connection is strongly supported, $M_{\text{soft},ij} \approx 1$, allowing full attention.
If a connection is unlikely, $M_{\text{soft},ij} \approx 0$, suppressing the attention.

## Pathways

A biological process or pathway can be thought as the set of concerted biochemical reactions
needed to perform a specific task within the cell [^1] [^2].
In the context of this work, we loosely identify a biological process or pathway as the genes contained within it, discarding information regarding other molecules or interactions.
We will therefore use the term pathway in the rest of the manuscript.
From this point of view, pathways can simply be thought of as gene sets, where these gene sets can overlap or even contain one another.

Selecting an appropriate set of pathways is crucial for our analyses.
The ideal selection should include all those pathways that are active in the system under study.
At the same time, it is desirable to reduce the redundancy that usually comes with large sets of pathways.
Following [^3], we considered the Gene Ontology pathways [^1], and selected pathways with less than 30 genes.
We then discarded those with more than half of their genes in common with other selected processes, as well as those with low replicability as defined in [^3].
We further refined this selection by including only those pathways that have at least five genes represented in our input datasets, and by removing those that are ancestors of other terms within our list.
This multi-step selection ensures that the final pathways are (mostly) non-overlapping and cover a large variety of processes in the system under study.

## References

[^1] [Ashburner et al. (2020)](https://doi.org/10.1038/75556)

[^2] [Kanehisa & Goto (2000)](https://doi.org/10.1093/nar/28.1.27)

[^3] [Ruiz-Arenas et al. (2024)](https://doi.org/10.1093/nar/gkae197)
