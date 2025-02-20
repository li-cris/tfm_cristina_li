from tqdm import tqdm


def train_taae(
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
        for (
            batch_gene_indices,
            batch_expression_values,
            batch_pathway_indices,
        ) in train_dataloader:  # noqa: N806
            batch_gene_indices, batch_expression_values = (
                batch_gene_indices.to(device),
                batch_expression_values.to(device),
            )  # noqa: N806
            optimizer.zero_grad()
            Y_predicted = model(  # noqa: N806
                batch_gene_indices, batch_expression_values, batch_pathway_indices
            )
            loss = criterion(Y_predicted, batch_expression_values)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        pbar.set_postfix_str(f"Training Loss: {avg_train_loss:.4f}")

    return model
