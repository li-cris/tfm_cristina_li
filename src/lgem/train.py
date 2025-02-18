from tqdm import tqdm


def train(model, criterion, optimizer, train_dataloader, n_epochs):
    model.train()

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for _ in pbar:
        epoch_loss = 0

        for batch_P, batch_Y in train_dataloader:  # noqa: N806
            optimizer.zero_grad()
            Y_predicted = model(batch_P)  # noqa: N806
            loss = criterion(Y_predicted, batch_Y.T)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        pbar.set_postfix_str(f"Loss: {avg_epoch_loss:.4f}")

    return model
