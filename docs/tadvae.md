# Transformer-Based Activity Discrepancy Variational Autoencoder

_Jan Voges,_
_Carlos Ruiz Arenas,_
_Jesús de la Fuente Cedeño,_
_Guillermo Dufort y Álvarez,_
_Álvaro Martín,_
_Idoia Ochoa,_
_and Mikel Hernaez_

## Introduction

One possibility to gain insights into how cells function is to systematically measure the transcriptional response to genetic perturbations.
While large-scale perturbation experiments are possible in cell biology, the sheer number of potential perturbations is overwhelming~\cite{uhler_building_2024}.
Considering that the human genome contains approximately 20,000 protein-coding genes~\cite{chi_dark_2016}, it is impractical to systematically perturb every possible multi-way combination of these genes across all relevant contexts such as cell types and diseases states.
Therefore, in-silico models are needed to predict the effects of untested perturbation/context combinations based on a limited set of observed data.

## Data

In general, \gls{scRNA-seq} data are processed into a gene expression matrix $\textbf{X} \in \mathbb{R}^{|\mathcal{C}| \times |\mathcal{G}_s|}$, where $|\mathcal{C}|$ is the number of cells and $|\mathcal{G}_s|$ is the number of measured genes.
Each element represents the read count of an RNA molecule for \gls{scRNA-seq} data or chromatin accessibility of a peak region for \gls{scATAC-seq} data.

To ensure accurate downstream analysis, further data processing is necessary.
The raw count matrix undergoes several preprocessing steps.

### Normalization

The raw counts are first normalized to account for varying sequencing depths across cells.
This results in normalized counts that allow comparisons across cells, even when sequencing depth varies significantly.

\Gls{CPM} normalization is a method used in RNA-seq data analysis to account for differences in sequencing depth across samples or cells.
It adjusts the raw read counts to make them comparable across different samples by scaling them relative to the total number of reads.
This allows for a fair comparison of gene expression levels across samples with varying sequencing depths.

For each gene in a sample, the raw count is divided by the total number of reads (or counts) in that sample, and then multiplied by one million (hence ``per million'').
The formula for calculating \gls{CPM} for a gene is:

\[
\text{CPM} = \frac{\text{Raw count}}{\text{Total counts in sample}} \cdot 10^6
\].

The raw count is the number of reads mapped to a particular gene in the sample.
The total counts in sample is the sum of all raw counts across all genes in that sample.
In \gls{CPM}, the scaling factor $10^6$ is applied to convert the result to ``per million''.

% Why CPM normalization is important:
RNA-seq experiments often yield different numbers of total reads for different samples, so \gls{CPM} normalization helps to correct for these differences, making it possible to compare gene expression across samples with varying sequencing depths.
Also, by scaling gene expression counts to per-million units, \gls{CPM} allows for fair comparisons of expression levels across samples, enabling more meaningful biological insights.

% Limitations of CPM normalization:
\gls{CPM} does not account for the length of the gene.
Longer genes tend to have more reads mapped to them simply due to their size, so methods like \gls{TPM} are sometimes preferred when gene length differences need to be accounted for.
Also, \gls{CPM} normalization may not perform well for lowly expressed genes, as the variance in these counts can be large and affect downstream analyses.

\subsubsection{Log Transformation}

After normalization, a log transformation is often applied to stabilize variance and reduce the impact of outliers, especially for highly expressed genes.
A log transformation stabilizes variance because it compresses the range of values in the data, particularly for high values, and reduces the influence of extreme outliers.

In many types of data, including gene expression counts, the variance tends to increase with the mean.
This is called heteroscedasticity, where the spread (variance) of data points is not uniform across the range of values.
By applying a log transformation, this relationship between the mean and variance is reduced, making the data more homoscedastic, which is important for many statistical models that assume constant variance.

In raw \gls{scRNA-seq} data, the variance of gene expression is often proportional to the mean expression level.
Genes with higher expression tend to have much larger variance compared to those with lower expression, making it hard to compare across genes.
For example, highly expressed genes can have very large count values and fluctuations, whereas lowly expressed genes tend to have smaller, more stable values.
The log transformation compresses the scale of the data by reducing the impact of high values and spreading out lower values.
Specifically:

\[
x_{\text{log-transformed}} = \log (x+1).
\]

The ``+1'' is added to avoid taking the log of zero.

When this transformation is applied, large numbers are squashed into a smaller range, and low values are spread out.
By compressing the large values and spreading out the smaller ones, the log transformation reduces the dependence of variance on the mean.

In addition to stabilizing variance, the log transformation also makes the data closer to a normal distribution, which is advantageous for many statistical techniques that rely on the assumption of normally distributed errors or residuals.
This transformation thus facilitates more reliable comparisons between genes and cells, allowing for better interpretation of biological signals.

\subsubsection{Highly Variable Gene Selection}

To focus on the most informative features, the data are often restricted to the most highly variable genes.
This step helps in reducing noise from genes that are consistently lowly expressed across cells or show little variation, which are less likely to contribute to distinguishing different cell types or states.
A common approach is to retain the top 2,000--5,000 most variable genes.

## Preliminaries and Background

### Causal Representation Learning

\Gls{CRL} has recently emerged as a promising approach for identifying the latent factors that \emph{causally} govern the systems under study~\cite{scholkopf_toward_2021,ahuja_interventional_2023}.
Among other disciplines, \gls{CRL} methods have been applied to biological systems, offering testable predictions about causal factors linked to diseases or treatment resistance~\cite{zhang_identifiability_2023,lopez_learning_2023}.

In the biological domain, data from Perturb-seq experiments have proven to be an ideal testbed for such analyses~\cite{dixit_perturb-seq_2016}.
This technology enables gene expression profiling of single cells in both their unperturbed state and when one or more genes functionally inactivated~\cite{gilbert_genome-scale_2014}.

Deep learning approaches have further advanced this field, enabling the prediction of transcriptional outcomes for novel perturbations of their combinations in Perturb-seq data~\cite{roohani_predicting_2024,cui_scgpt_2024}, as well as known perturbations in previously uncharacterized cell types~\cite{lotfollahi_scgen_2019}.

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

## Tokenization and Embedding of Gene Expression Data

Each gene expression profile is a vector where each dimension corresponds to a specific gene, and each value is the normalized expression level of that gene.
This setting differs fundamentally from e.g., text input, where tokenization and embeddings operate on discrete symbols (words, subword, or characters).

We perform the following steps to map the gene expression profiles to a Transformer-friendly input format.

**Step 1: Treat each gene as a token.**
Each gene is naturally a token.
So, for an input gene expression profile with $n$ genes, we already have $n$ tokens.

**Step 2: Use a learnable gene embedding matrix.**
We use a learnable embedding table for genes:

$$
E \in \mathbb{R}^{n_{\text{genes}} \times d_{\text{embed}}}
$$

where $n_{\text{genes}}$ is the total number of genes across datasets, and $d_{\text{embed}}$ is the embedding dimension.
Each row in $E$ represents a learnable embedding vector for a gene.

The number of possible genes $n_{\text{genes}}$ can be either set to e.g., the total number of protein-coding genes (\~20,000), or to the cardinality of the union of all genes across datasets.

For scRNA-seq datasets with \~5,000 genes per sample, we choose $d_{\text{embed}} = 128$ or $d_{\text{embed}} = 256$.

> Note: In a multi-head attention mechanism, the input embedding dimension is split across multiple attention heads.
> If there are $h$ heads, each head gets a feature dimension of $d_{\text{head}}$, defined as:
>
> $$
> d_{\text{head}} = \frac{d_{\text{model}}}{h}
> $$
>
> Therefore, for $d_{\text{embed}} = 128$, we chose $h = 8$ (so $d_{\text{head}} = 16$).
> For $d_{\text{embed}} = 256$, we chose $h = 16$ (so $d_{\text{head}} = 16$).

**Step 3: Modulate the embeddings with expression values.**
We weight the gene embeddings by their respective expression values:

$$
X_{\text{input}} = E \odot X
$$

where $X \in \mathbb{R}^{n_{\text{genes}}}$ is the gene expression profile (the raw input) and $\odot$ is the element-wise multiplication.
The result, $X_{\text{input}}$, is a matrix where each row represents a gene's embedding scaled by its expression value.
This means that if a gene is highly expressed, its embedding contributes more, and accordingly, if a gene has low or zero expression, its embedding has little or no impact.

## Embedding of Pathways

TODO

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

We initialize $W_{\text{mask}}$ as:

$$
W_{\text{mask}} = M + \epsilon
$$

where $\epsilon$ is a small random perturbation from a normal distribution to allow the model to adjust connections.
As $M$ encodes prior biological knowledge about which genes belong to which pathways, we bias the model towards known relationships at the start of training.
The model can then fine-tune these relationships based on data, rather than learning them from scratch.

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

[^1]: [Ashburner et al. (2020)](https://doi.org/10.1038/75556)

[^2]: [Kanehisa & Goto (2000)](https://doi.org/10.1093/nar/28.1.27)

[^3]: [Ruiz-Arenas et al. (2024)](https://doi.org/10.1093/nar/gkae197)
