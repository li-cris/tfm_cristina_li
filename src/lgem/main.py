import os

import torch
import torch.nn as nn

from .data import compute_embeddings, create_dataloaders, load_data
from .models import LinearGeneExpressionModelLearned, LinearGeneExpressionModelOptimized
from .test import test
from .train import train
from .utils import get_git_root


def main():
    torch.manual_seed(42)

    dataset_name = "NormanWeissman2019_filtered"
    Y_train, perturbations, genes = load_data(dataset_name)  # noqa: N806

    G, P, b = compute_embeddings(Y_train, perturbations, genes)  # noqa: N806

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        Y_train, P, batch_size=32
    )

    # Set up the learned model.
    model = LinearGeneExpressionModelLearned(G, b)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    n_epochs = 5000
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter '{name}' will be updated during training.")

    # Train the learned model and save it.
    artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "lgem", dataset_name)
    model_file_path = os.path.join(artifacts_dir_path, "model.pt")
    if False:
        model = train(model, criterion, optimizer, train_dataloader, n_epochs)
        torch.save(model.state_dict(), model_file_path)

    # Load the trained model and test it.
    model_trained = LinearGeneExpressionModelLearned(G, b)
    model_trained.load_state_dict(torch.load(model_file_path))
    test(model_trained, criterion, test_dataloader)

    # Fit the optimized model and test it.
    model_optimized = LinearGeneExpressionModelOptimized(Y_train, G, P, b)
    test(model_optimized, criterion, test_dataloader)

    # Compare the learned and optimized models.
    W_trained = model_trained.W.detach()  # noqa: N806
    W_optimized = model_optimized.W.detach()  # noqa: N806
    mse = torch.mean((W_trained - W_optimized) ** 2).item()
    print(f"Mean squared error between trained and optimized W: {mse:.4f}")


if __name__ == "__main__":
    main()
