import torch


def test(model, criterion, test_dataloader):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_P, batch_Y in test_dataloader:  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            loss = criterion(Y_predicted, batch_Y.T)
            test_loss += loss.item()
            num_batches += 1

    avg_test_loss = test_loss / num_batches
    print(f"Average test loss: {avg_test_loss:.4f}")
