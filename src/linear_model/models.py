import torch  # noqa: D100
import torch.nn as nn


class LinearGeneExpressionModel(nn.Module):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        G,  # (n_genes, K)  # noqa: N803
        b,  # (n_genes)
    ):
        super().__init__()

        self.G = G
        self.W = None
        self.b = b

    def forward(self, P):  # noqa: D102, N803
        return self.G @ self.W @ P.T + self.b.unsqueeze(1)


class LinearGeneExpressionModelTrained(LinearGeneExpressionModel):  # noqa: D101
    def __init__(self, G, b):  # noqa: D107, N803
        super().__init__(G, b)

        # Set up W as learnable parameter.
        d_embed = G.shape[1]
        self.W = nn.Parameter(data=torch.randn(d_embed, d_embed))


class LinearGeneExpressionModelOptimized(LinearGeneExpressionModel):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        Y_train,  # (n_genes, n_perturbations)  # noqa: N803
        G,  # (n_genes, d_embed)  # noqa: N803
        P,  # (n_perturbations, d_embed)  # noqa: N803
        b,  # (n_genes)
    ):
        super().__init__(G, b)

        # Fit W by minimizing argmin_W ||Y - (GWP^T + b)||^2.
        Y_centered = Y_train - b.unsqueeze(1)  # (n_genes, n_perturbations)  # noqa: N806
        Y_vec = Y_centered.flatten()  # (n_genes * n_perturbations)  # noqa: N806
        Kron_P_G = torch.kron(P, G)  # (n_perturbations * n_genes, d_embed * d_embed)  # noqa: N806
        W_vec, _, _, _ = torch.linalg.lstsq(Kron_P_G, Y_vec)  # (d_embed * d_embed)  # noqa: N806
        d_embed = G.shape[1]
        self.W = W_vec.reshape(d_embed, d_embed)  # (d_embed, d_embed)
