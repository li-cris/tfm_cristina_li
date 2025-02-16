import torch  # noqa: D100


def test(model, criterion, test_dataloader):  # noqa: D103
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_P, batch_Y in test_dataloader:  # noqa: N806
            Y_pred = model(batch_P)  # noqa: N806
            loss = criterion(Y_pred, batch_Y.T)
            print(f"Test loss: {loss.item():.4f}")
            test_loss += loss.item()
            num_batches += 1

    avg_test_loss = test_loss / num_batches
    print(f"Average test loss: {avg_test_loss:.4f}")
