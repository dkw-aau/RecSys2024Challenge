import torch


def normalize(embs):
    max_c = torch.max(embs, dim=0, keepdim=True)[0]
    min_c = torch.min(embs, dim=0, keepdim=True)[0]
    return (embs - min_c) / (max_c - min_c)


def standarize(embs, mean=None, std=None):
    if mean is None:
        mean = torch.mean(embs, dim=0, keepdim=True)
        std = torch.std(embs, dim=0, keepdim=True)
    elif std is None:
        std = torch.std(mean, dim=0, keepdim=True)
        mean = torch.mean(mean, dim=0, keepdim=True)
    return (embs - mean) / (2 * std)