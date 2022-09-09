import torch
import numpy as np
from torch import nn, optim
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from model.module import FM, PNN, PredictMLP


class PredictModel(nn.Module):
    '''
    A base class of all available predict models.
    '''
    def __init__(self, config):
        super().__init__()
        self.__dict__.update(config)
        self.W1 = nn.Embedding.from_pretrained(torch.normal(0, 1e-3, (self.max_features, 1), requires_grad=True))
        self.V = nn.Embedding.from_pretrained(torch.normal(0, 1e-3, (self.max_features, self.embedding_size), requires_grad=True))
        
        self.model = self.get_model()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.7 ** (1/self.decay_step))
        self.writer = SummaryWriter()

    def get_model(self):
        model = nn.Sequential()
        if self.model_name == "FM":
            model.add_module("FM", FM(self.W1, self.V, point_wise=True))
        elif self.model_name == "PNN":
            model.add_module("PNN", PNN(self.W1, self.V))
            model.add_module("PredictMLP", PredictMLP(2 * self.embedding_size, self.prediction_hidden_width, self.keep_prob))
            model.add_module("Linear", nn.Linear(self.prediction_hidden_width[-1], 1))
        return model

    def get_loss_and_metrics(self, y, y_hat, is_training):
        y_hat = torch.sigmoid(y_hat)
        base_loss_fn = nn.BCELoss(reduction="mean")
        base_loss = base_loss_fn(y_hat, y)
        loss = base_loss
        for name, param in self.model.named_parameters():
            if not (name.endswith("bias") or "bn" in name): # only consider W
                loss += torch.sum(param.pow(2)) / 2
        
        threshold = 0.5
        ones, zeros = torch.ones_like(y), torch.zeros_like(y)
        y_hat_int = np.where(y_hat >= threshold, ones, zeros)
        eval_metrics = {
            "auc": roc_auc_score(y, y_hat), # what about the shape of y?
            "acc": np.sum(y_hat_int == y) / len(y), # whether to add "axis"?
        }
        return base_loss, loss, eval_metrics

