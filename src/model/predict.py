import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

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
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.l2_reg)
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

    def forward(self, input):
        return self.model(input)