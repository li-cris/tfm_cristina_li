import os  # noqa: D100

import torch
import torch.nn as nn

from .data import compute_embeddings, create_dataloaders, load_data
from .models import LinearGeneExpressionModelTrained
from .test import test
from .train import train
from .utils import get_git_root

Y_train, perturbations, genes = load_data(dataset_name="NormanWeissman2019_filtered")
G, P, b = compute_embeddings(Y_train=Y_train, perturbations=perturbations, genes=genes, d_embed=10)
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(Y_train, P, batch_size=32)

torch.manual_seed(seed=42)
model = LinearGeneExpressionModelTrained(G, b)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter '{name}' will be updated during training.")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
n_epochs = 5000

# Train the model and save it.
model = train(model, criterion, optimizer, train_dataloader, n_epochs)
artifacts_dir_path = os.path.join(get_git_root(), "artifacts", "linear_model")
model_file_path = os.path.join(artifacts_dir_path, "model.pth")
torch.save(model.state_dict(), model_file_path)

# Load the trained model and test it.
model_trained = LinearGeneExpressionModelTrained(G, b)
model_trained.load_state_dict(torch.load(model_file_path))
test(model_trained, criterion, test_dataloader)

# # Compare the trained model to the optimized model.
# model_optimized = LinearGeneExpressionModelOptimized(Y_train=Y_train.T, G=G, P=P, b=b)
# W_trained = model_trained.W.detach()
# W_optimized = model_optimized.W.detach()
# mse = torch.mean((W_trained - W_optimized) ** 2).item()
# print(f"Mean squared error between trained and optimized W: {mse:.4f}")
