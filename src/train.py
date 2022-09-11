import torch
import numpy as np
from torch import nn
from tqdm import trange
from sklearn.metrics import roc_auc_score


def eval(y, y_hat):
    threshold = 0.5
    ones, zeros = torch.ones_like(y), torch.zeros_like(y)
    y_hat_int = torch.Tensor(np.where(y_hat >= threshold, ones, zeros))
    eval_metrics = {
        "auc": roc_auc_score(y, y_hat.detach()),
        "acc": (torch.sum(y_hat_int == y) / y.numel()).item(),
    }
    return eval_metrics


def train(model, train_iter, epochs, device):
    for epoch in trange(epochs):
        train_loss, num_samples = 0.0, 0
        for batch, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = torch.sigmoid(y_hat)
            loss = nn.BCELoss(reduction="mean")(y_hat, y)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            train_loss += loss * y.numel()
            num_samples += y.numel()
        model.writer.add_scalar("train_loss", train_loss / num_samples, epoch)

