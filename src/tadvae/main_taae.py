import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_taae import load_data
from .models.taae import TAAE
from .train_taae import train_taae
from .utils import get_git_root


def main():
    dataset_name = "norman"

    artifacts_dir_path = os.path.join(
        get_git_root(), "artifacts", "tadvae", dataset_name
    )
    adata_file_path = os.path.join(artifacts_dir_path, "adata.pt")
    gpmask_file_path = os.path.join(artifacts_dir_path, "gpmask.pt")
    if os.path.exists(adata_file_path) and os.path.exists(gpmask_file_path):
        adata = torch.load(adata_file_path, weights_only=False)
        gpmask = torch.load(gpmask_file_path, weights_only=False)
    else:
        adata, gpmask = load_data(dataset_name)
        os.makedirs(artifacts_dir_path, exist_ok=True)
        torch.save(adata, adata_file_path)
        torch.save(gpmask, gpmask_file_path)

    # TODO: Min-max normalization needed for expression values?
    # TODO: Also get gene and pathway Ensembl IDs.

    n_genes = adata.shape[1]
    n_pathways = gpmask.shape[1]
    d_embed = 32
    n_heads = 2
    n_layers = 4
    d_hidden = 32
    batch_size = 16

    W_init = torch.randint(2, (n_genes, n_pathways))  # noqa: N806

    # Convert expression values to tensor.
    expression_values = torch.tensor(adata.X.todense()).float()
    n_samples = expression_values.shape[0]

    # Get gene indices for all adata.var_names. Repeat for all samples.
    gene_indices = torch.arange(adata.shape[1])
    gene_indices = gene_indices.unsqueeze(0).repeat(n_samples, 1)

    # Get pathway indices from gene_pathway_mask. Repeat for all samples.
    pathway_indices = torch.arange(n_pathways)
    pathway_indices = pathway_indices.unsqueeze(0).repeat(n_samples, 1)

    print(f"n_samples: {n_samples}")
    print(f"n_genes: {n_genes}")
    print(f"n_pathways: {n_pathways}")

    train_dataset = TensorDataset(gene_indices, expression_values, pathway_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TAAE(
        n_genes,
        n_pathways,
        d_embed,
        n_heads,
        n_layers,
        d_hidden,
        mask_type="soft",
        W_init=W_init,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    n_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_taae(model, criterion, optimizer, train_dataloader, n_epochs, device)


if __name__ == "__main__":
    main()
