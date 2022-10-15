import torch


def sigmoid_deriv(x):
    y = torch.sigmoid(x)
    return y * (1 - y)


def tanh_deriv(x):
    y = torch.tanh(x)
    return 1 - y**2


def identity(x):
    return x


def identity_deriv(x):
    return torch.ones_like(x)
