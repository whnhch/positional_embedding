import torch
import numpy as np


def get_positional_embedding(max_len, dim=768):
    pe = torch.Tensor(max_len, dim)

    for i in range(max_len):
        pe[i, :] = (i + 1) / dim

    return pe


def get_binary_embedding(max_len, dim):
    pe = torch.Tensor(max_len, dim)
    bin_form = '{0:' + str(max_len) + 'b}'

    for i in range(max_len):
        bin_form.format(i)
        pe[i, :] = int(bin_form)

    return pe


def get_sinusoidal_embedding(max_len, dim=768):
    pe = torch.Tensor(max_len, dim)

    pos = torch.arange(0, max_len, 1.).unsqueeze(1)
    k = torch.exp(-np.log(10000) * torch.arrange(0, dim, 2.) / dim)

    pe[:, 0::2] = torch.sin(pos * k)
    pe[:, 1::2] = torch.cos(pos * k)

    return pe
