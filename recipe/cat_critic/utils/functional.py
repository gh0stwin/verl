import torch


def symlog(x, mean, var):
    x = x * var + mean
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x, mean, var):
    x = torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
    return x * var + mean
