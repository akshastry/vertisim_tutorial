import torch


def min_max_normalize(tensor, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val, _ = torch.min(tensor, dim=0)
        max_val, _ = torch.max(tensor, dim=0)
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def standardize(tensor):
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)
    standardized = (tensor - mean) / std
    return standardized, mean, std