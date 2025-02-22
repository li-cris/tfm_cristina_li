import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .data import compute_embeddings, load_pseudobulk_data
from .models import (
    LinearGeneExpressionModelLearned,
    LinearGeneExpressionModelOptimized,
)
from .test import test
from .train import train
from .utils import get_git_root

torch.serialization.add_safe_globals({"list": list})


def main():
    # Parameters.
    batch_size = 8
    criterion = nn.MSELoss()
    n_epochs = 20000

    # PyTorch setup.
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the data.
    dataset_name = "NormanWeissman2019_filtered"
    Y, perts, genes = load_pseudobulk_data(dataset_name)  # noqa: N806

    # Compute the embeddings on the entire dataset.
    G, P, b = compute_embeddings(Y.T, perts, genes)  # noqa: N806

    # Split the data into training and test sets and create the dataloaders.
    Y_train, Y_test, P_train, P_test = train_test_split(  # noqa: N806
        Y, P, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(P_train, Y_train)
    test_dataset = TensorDataset(P_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Fit the optimized model to the training data.
    model_optimized = LinearGeneExpressionModelOptimized(Y_train.T, G, P_train, b)

    # Test the optimized model.
    test_loss = test(model_optimized, criterion, test_dataloader, device)
    print(f"Test loss (optimized model): {test_loss:.4f}")

    # Fit the learned model to the training data.
    model_learned = LinearGeneExpressionModelLearned(G, b)
    optimizer = torch.optim.Adam(params=model_learned.parameters(), lr=1e-3)
    model_learned = train(
        model_learned, criterion, optimizer, train_dataloader, n_epochs, device
    )

    # Test the learned model.
    test_loss = test(model_learned, criterion, test_dataloader, device)
    print(f"Test loss (learned model): {test_loss:.4f}")


if __name__ == "__main__":
    main()
