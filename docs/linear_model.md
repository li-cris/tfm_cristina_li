# The Linear Gene Expression Model

In the Linear Gene Expression Model (LGEM) by [Ahlmann-Eltze et al. (2024)](https://doi.org/10.1101/2024.09.16.613342), we have:
- Embeddings for read-out genes: $G$ (an $n \times d_{\text{embed}}$ matrix, where $n$ is the number of read-out genes and $d_{\text{embed}}$ is the dimensionality of the embeddings).
- Embeddings for perturbations: $P$ (an $m \times d_{\text{embed}}$ matrix, where $m$ is the number of perturbations).

Given a data matrix $Y_{\text{train}}$ of gene expression values, the model then fits the matrix $W$ by minimizing:

$$
\arg \min_{W} \| Y_{\text{train}} - (G W P^\top + b) \|^2
$$

Hence, we furthermore have:
- Data matrix: $Y_{\text{train}}$ (an $n \times m$ matrix, i.e., pseudobulked per condition/perturbation).
- Weight matrix: $W$ (a $d_{\text{embed}} \times d_{\text{embed}}$ matrix to be learned).
- Bias vector: $b$ (an $n \times 1$ vector of average gene expressions).

The model then predicts gene expression values using:

$$
Y_{\text{train}} \approx G W P^\top + b
$$

Note: The bias vector $b$ (with dimensions $n \times 1$) is added to each column of the matrix $G W P^\top$ (with dimensions $n \times m$).
This operation effectively _broadcasts_ the vector $b$ across all $m$ columns, repeating it $m$ times to match the dimensions of $G W P^\top$.
Note that e.g., PyTorch handles broadcasting automatically implicitly.

In the paper, to obtain the embeddings $G$ and $P$, they performed a PCA on $Y_{\text{train}}$ and used the top $d_{\text{embed}}$ principal components for $G$.
They then subset this $G$ to only the rows corresponding to genes that have been perturbed in the training data (and hence appear as columns in $Y_{\text{train}}$) and used the resulting matrix for $P$.
When using the model for prediction, they replace $P$ with the matrix formed by the rows of $G$ corresponding to genes perturbed in the test data.

We can vectorize the equation $Y_{\text{centered}} = G W P^\top$ and set it up in a form suitable for least squares.

Using the mixed-product property of the Kronecker product, we have:

$$
\text{vec} (Y_{\text{centered}}) = (P \otimes G) \text{vec} (W)
$$

Here:
- $\text{vec} (Y_{\text{centered}})$ is the vectorization of $Y_{\text{centered}}$ (flattened column-wise).
- $P \otimes G$ is the Kronecker product of $P$ and $G$.
- $\text{vec} (W)$ is the vectorization of $W$ (flattened column-wise).

We can now solve the linear equation:

$$
Y_{\text{vec}} = (P \otimes G) \cdot W_{\text{vec}}
$$

# Formulating the Linear Gene Expression Model As a Neural Network

The LGEM formulation matches the standard linear layer in neural networks, where the output is a linear transformation of the input plus a bias term.
By combining $G$ and $W$ into a single matrix $M = G W$ with dimensions $n \times d_{\text{embed}}$, we can write the prediction for each perturbation as:
$$
y = M p^\top + b
$$

Here:
- $y$ is the predicted gene expression vector ($n \times 1$).
- $p^\top$ is the transpose of the perturbation embedding vector ($d_{\text{embed}} \times 1$).
- $M$ serves as the weight matrix in the neural network.
- $b$ is the bias vector.

This can directly be interpreted as the standard linear layer given by:
$$
y = W x + b
$$

We now choose to keep $G$ fixed (i.e., it consists of the top $d_{\text{embed}}$ principal components from a PCA on $Y_{\text{train}}$), and only learn $W$.
Then, the neural network (_without activation functions_) is equivalent to the original LGEM.
$M = G W$ would then be a combination of the fixed $G$ and the learned $W$.
