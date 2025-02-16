from tqdm import tqdm  # noqa: D100


def train(model, criterion, optimizer, train_dataloader, n_epochs):  # noqa: D103
    model.train()

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for _ in pbar:
        epoch_loss = 0

        for batch_P, batch_Y in train_dataloader:  # noqa: N806
            optimizer.zero_grad()
            Y_pred = model(batch_P)  # noqa: N806
            loss = criterion(Y_pred, batch_Y.T)  # Transpose back to (n_genes x batch_size).
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        pbar.set_postfix_str(f"Loss: {avg_epoch_loss:.4f}")

    return model
