from tqdm import tqdm
import torch


def train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    validation_dataloader = None,
    n_epochs: int = 10,
    device: str  = "cuda",
    validation: bool = True
):
    model.to(device)
    best_model = model.state_dict()

    pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    best_val_loss = 2000.0

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
        postfix = f"Training Loss: {avg_train_loss:.4f}"

        if validation and validation_dataloader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_batch_P, val_batch_Y in validation_dataloader:
                    val_batch_P, val_batch_Y = val_batch_P.to(device), val_batch_Y.to(device)
                    val_Y_predicted = model(val_batch_P)
                    val_loss += criterion(val_Y_predicted, val_batch_Y.T).item()

                avg_val_loss = val_loss / len(validation_dataloader)
                postfix += f" | Validation Loss: {avg_val_loss:.4f}"

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = model.state_dict()

        pbar.set_postfix_str(postfix)

    if validation and validation_dataloader is not None:
        model.load_state_dict(best_model)

    return model
