# Linear Gene Expression Model

## Mathematical Formulation of the Linear Gene Expression Model

In the Linear Gene Expression Model (LGEM) [^1], we have:
- Gene embedding matrix: $G$ (an $n_{\text{genes}} \times d_{\text{embed}}$ matrix).
- Perturbation embedding matrix: $P$ (an $n_{\text{perturbations}} \times d_{\text{embed}}$ matrix).

Given a data matrix $Y_{\text{train}}$ of gene expression values, the model then fits the matrix $W$ by minimizing:

$$
\arg \min_{W} \| Y_{\text{train}} - (G W P^\top + b) \|^2
$$

Here:
- Data matrix: $Y_{\text{train}}$ (an $n_{\text{genes}} \times n_{\text{perturbations}}$ matrix, i.e., pseudobulked per condition/perturbation).
- Weight matrix: $W$ (a $d_{\text{embed}} \times d_{\text{embed}}$ matrix to be fitted).
- Bias vector: $b$ (an vector with $n_{\text{genes}}$ elements of average gene expression values).

The model then predicts gene expression values using:

$$
Y_{\text{predicted}} = G W P^\top + b
$$

Here, $P$ is the matrix of perturbations to be predicted.
It is formed by those rows of $G$ correponding to the perturbed genes.

Note: The bias vector $b$ is added to each column of the matrix $G W P^\top$.
This operation effectively _broadcasts_ $b$ across all $n_{\text{perturbations}}$ columns of $G W P^\top$.

In [^1], to obtain the embeddings $G$ and $P$, they performed a principal component analysis (PCA) on $Y_{\text{train}}$ and used the top $d_{\text{embed}}$ principal components for $G$.
They then subset this $G$ to only those rows corresponding to genes that have been perturbed in the training data (and hence appear as columns in $Y_{\text{train}}$) and used the resulting matrix for $P$.

To fit the weight matrix $W$, we vectorize the equation $Y_{\text{centered}} = Y_{\text{train}} - b = G W P^\top$ and set it up in a form suitable for least squares.

Using the mixed-product property of the Kronecker product, we have:

$$
\text{vec} (Y_{\text{centered}}) = (P \otimes G) \text{vec} (W)
$$

Here:
- $\text{vec} (Y_{\text{centered}})$ is the vectorization of $Y_{\text{centered}}$ (flattened column-wise).
- $P \otimes G$ is the Kronecker product of $P$ and $G$.
- $\text{vec} (W)$ is the vectorization of $W$ (flattened column-wise).

We can now solve this linear equation.

## Formulating the Linear Gene Expression Model As a Neural Network

The LGEM formulation matches the standard linear layer in neural networks, where the output is a linear transformation of the input plus a bias term.
By combining $G$ and $W$ into a single matrix $M = G W$ with dimensions $n_{\text{genes}} \times d_{\text{embed}}$, we can write the prediction for a perturbation as:

$$
Y_{\text{predicted}} = M P^\top + b
$$

This can directly be interpreted as a standard linear layer, where $M$ serves as the weight matrix.

We now choose to keep $G$ fixed (i.e., it consists of the top $d_{\text{embed}}$ principal components from a PCA on $Y_{\text{train}}$), and only learn $W$.
Then, the neural network (_without activation functions_) is equivalent to the original LGEM.

## References

[^1]: [Ahlmann-Eltze et al. (2024)](https://doi.org/10.1101/2024.09.16.613342)
