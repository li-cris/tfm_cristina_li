import torch


def test(model, criterion, test_dataloader, device):
    model.to(device)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_P, batch_Y in test_dataloader:  # noqa: N806
            batch_P, batch_Y = batch_P.to(device), batch_Y.to(device)  # noqa: N806
            Y_predicted = model(batch_P)  # noqa: N806
            loss = criterion(Y_predicted, batch_Y.T)
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_dataloader)

    return avg_test_loss
