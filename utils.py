# utils.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_similarity(x, y, dim=1, eps=1e-8):
    """Computes cosine similarity along a specified dimension."""
    w12 = torch.sum(x * y, dim)
    w1 = torch.norm(x, 2, dim)
    w2 = torch.norm(y, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def pearson_correlation(x, y):
    """ Computes Pearson correlation coefficient. Assumes x, y are [N, FeatureDim]"""
    vx = x - torch.mean(x, dim=0, keepdim=True)
    vy = y - torch.mean(y, dim=0, keepdim=True)
    r = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx**2, dim=1)) * torch.sqrt(torch.sum(vy**2, dim=1)) + 1e-8)
    # For single feature vectors (N=1), manually compute or handle NaN
    if r.numel() == 1 and torch.isnan(r):
        return torch.tensor(0.0, device=x.device) # Or handle as appropriate
    return r

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """Calculates AUC, ACC, SEN, SPE."""
    y_pred_label = (y_pred_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_label)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()

    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity (Recall)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity

    return {'AUC': auc, 'ACC': acc, 'SEN': sen, 'SPE': spe}

def build_mlp(dims, activation=torch.nn.ReLU, last_activation=None):
    """Builds a Multi-Layer Perceptron."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2: # No activation before final output layer unless specified
             if activation: layers.append(activation())
        elif last_activation: # Apply activation to the last layer if specified
             layers.append(last_activation())
    return torch.nn.Sequential(*layers)

# Add functions for saving/loading models, logging, etc.