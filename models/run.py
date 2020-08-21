import torch

from models import data
from models import networks


class TSPModel(object):
    param_dim = 2

    hidden_size = 128
    encoder_rnn_layers = 2
    decoder_rnn_layers = 2
    encoder_dropout = 0.1
    decoder_dropout = 0.1

    optimizer_object = torch.optim.SGD
    learning_rate = 0.01
    optimizer_kwargs = {}

    def __init__(self, **kwargs):
        for k, v in kwargs:
            self.__setattr__(k, v)

        self.network = networks.PointerNetwork(
            self.param_dim, self.hidden_size, self.encoder_rnn_layers, self.decoder_rnn_layers,
            encoder_dropout=self.encoder_dropout, decoder_dropout=self.decoder_dropout)
        self.optimizer = self.optimizer_object(self.network.parameters(), self.learning_rate, **self.optimizer_kwargs)

    def train(self, data_loader):
        self.network.train()
        self.optimizer.zero_grad()

    def train_step(self):
        self.optimizer.zero_grad()


