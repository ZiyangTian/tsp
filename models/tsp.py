import tqdm
import torch

from models import attentions
from models import data
from models import networks
from models import losses


class TSPModel(object):
    param_dim = 2

    hidden_size = 128
    rnn_layers = 2
    encoder_dropout = 0.
    encoder_rnn_type = torch.nn.LSTM
    encoder_rnn_dropout = 0.
    decoder_dropout = 0.
    decoder_rnn_type = torch.nn.LSTM
    decoder_rnn_dropout = 0.
    attention_mechanism = attentions.BahdanauAttention

    optimizer_object = torch.optim.Adam
    learning_rate = 0.01
    optimizer_kwargs = {}

    train_pattern = None
    train_batch_size = 128

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        encoder_rnn = self.encoder_rnn_type(
            self.hidden_size, self.hidden_size, self.rnn_layers,
            dropout=self.encoder_rnn_dropout, bidirectional=False)
        encoder = networks.PointerEncoder(self.param_dim, encoder_rnn, dropout=self.encoder_dropout)
        decoder_rnn = self.decoder_rnn_type(
            self.hidden_size, self.hidden_size, self.rnn_layers,
            dropout=self.decoder_rnn_dropout, bidirectional=False)
        attention = self.attention_mechanism(self.hidden_size)
        decoder = networks.PointerDecoder(self.param_dim, decoder_rnn, attention, dropout=self.decoder_dropout)

        self.network = networks.PointerNetwork(encoder, decoder)
        self.optimizer = self.optimizer_object(self.network.parameters(), self.learning_rate, **self.optimizer_kwargs)
        self.criterion = losses.TSPLoss()
        self.train_data_loader = data.TSPDataLoader(
            self.train_pattern, shift_rank_randomly=False, batch_size=self.train_batch_size, shuffle=False)

    def train(self, num_epochs):
        self.network.train()
        self.optimizer.zero_grad()

        for t in range(num_epochs):
            for padded_parameters, padded_ranks, lengths in self.train_data_loader:
                results = self.train_step(padded_parameters, padded_ranks, lengths)
                print(results['loss'])

    def train_step(self, padded_parameters, padded_ranks, lengths):
        output_scores, ranks = self.network(padded_parameters, lengths, padded_ranks)
        loss = self.criterion(output_scores, ranks, lengths)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'loss': loss.item(),
        }


def main():
    pattern = r'/Users/Tianziyang/Desktop/data/tsp/*'
    # dl = data.TSPDataLoader(pattern, batch_size=5)
    # for parameters, rank, lengths in dl:
    #     break
    # print(parameters.shape, rank.shape, lengths.shape)
#
    # model = networks.PointerNetwork.from_config({'input_size': 2, 'hidden_size': 8})
    # output_scores, ranks = model(parameters, lengths)
    # loss = losses.tsp_loss_fn(output_scores, rank, lengths)
    # print(output_scores.shape, ranks.shape, loss)
    model = TSPModel(train_pattern=pattern)
    model.train(100)


if __name__ == '__main__':
    main()
