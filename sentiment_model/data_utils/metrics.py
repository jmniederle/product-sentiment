import torch


def accuracy(model_output, labels):
    predicted = torch.argmax(model_output, dim=1)
    return torch.mean((predicted == labels).float())


def MAE(model_output, labels, apply_sigmoid=False):
    if apply_sigmoid:
        model_output = torch.sigmoid(model_output)

    return torch.mean(torch.abs(model_output - labels))
