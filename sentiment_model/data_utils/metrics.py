import torch


def accuracy(model_output, labels):
    predicted = torch.argmax(model_output, dim=1)
    return torch.mean((predicted == labels).float())
