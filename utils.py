import numpy as np
import torch


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()


def get_unk_score(K_prob, K_1_prob):
    entropy = torch.sum(- K_prob * torch.log(K_prob + 1e-10), dim=1, keepdim=True)
    entropy_norm = (entropy / np.log(K_prob.size(1))).squeeze(1)
    prod_max = torch.max(K_1_prob, dim=1)[0]

    unk_score = entropy_norm - prod_max  # entropy - confidence
    unk_score = unk_score.detach()
    return normalize_weight(unk_score)


def acc(y_pred,y_true):
    correct_predictions = (y_pred == y_true).sum().item()
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


