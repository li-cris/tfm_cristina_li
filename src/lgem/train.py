from tqdm import tqdm


def train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    n_epochs,
    device,
):
    model.to(device)

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")

    for _ in pbar:
        model.train()
        train_loss = 0.0
        for batch_P, batch_Y in train_dataloader:  # noqa: N806
            batch_P, batch_Y = batch_P.to(device), batch_Y.to(device)  # noqa: N806
            optimizer.zero_grad()
            Y_predicted = model(batch_P)  # noqa: N806
            loss = criterion(Y_predicted, batch_Y.T)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        pbar.set_postfix_str(f"Training Loss: {avg_train_loss:.4f}")

    return model
