import torch
import torch.nn as nn


class LinearGeneExpressionModelLearned(nn.Module):
    def __init__(self, G: torch.Tensor, b: torch.Tensor) -> None:  # noqa: N803
        """Linear gene expression model with learnable parameters.

        Args:
            G: Gene embedding matrix with shape (n_genes, d_embed).
            b: Bias vector with shape (n_genes).
        """
        super().__init__()

        _, d_embed = G.shape

        self.register_buffer("G", G)
        self.register_buffer("b", b)
        self.W = nn.Parameter(data=torch.randn((d_embed, d_embed)))

    def forward(self, P: torch.Tensor) -> torch.Tensor:  # noqa: N803
        M = self.G @ self.W  # noqa: N806
        return M @ P.T + self.b.unsqueeze(1)


class LinearGeneExpressionModelOptimized(nn.Module):
    def __init__(
        self,
        Y_train: torch.Tensor,  # noqa: N803
        G: torch.Tensor,  # noqa: N803
        P: torch.Tensor,  # noqa: N803
        b: torch.Tensor,
    ) -> None:
        """Linear gene expression model, optimized using a least squared problem solver.

        Args:
            Y_train: Data matrix with shape (n_genes, n_perturbations).
            G: Gene embedding matrix with shape (n_genes, d_embed).
            P: Perturbation embedding matrix with shape (n_perturbations, d_embed).
            b: Bias vector with shape (n_genes).
        """
        super().__init__()

        _, d_embed = G.shape

        self.register_buffer("G", G)
        self.register_buffer("b", b)

        Y_centered = Y_train - b.unsqueeze(1)  # noqa: N806
        Y_vec = Y_centered.flatten()  # noqa: N806
        Kron_P_G = torch.kron(P, G)  # noqa: N806
        W_vec, _, _, _ = torch.linalg.lstsq(Kron_P_G, Y_vec)  # noqa: N806

        self.register_buffer("W", W_vec.reshape((d_embed, d_embed)))

    def forward(self, P: torch.Tensor) -> torch.Tensor:  # noqa: N803
        return self.G @ self.W @ P.T + self.b.unsqueeze(1)
