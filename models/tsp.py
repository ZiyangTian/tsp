import tqdm
import torch

from models import attentions
from models import losses
from models import metrics
from models import networks


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

    optimizer_obj = torch.optim.Adam
    learning_rate = 0.01
    optimizer_kwargs = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.network = self._build_network()
        self.optimizer = self.optimizer_obj(self.network.parameters(), self.learning_rate, **self.optimizer_kwargs)
        self.loss_fn = losses.TSPLoss()
        self.metric_fns = {
            'val_loss': metrics.TSPLoss(),
            'bidirectional_accuracy': metrics.BidirectionalAccuracy(),
            'journey_mae': metrics.JourneyMAE(),
            'journey_mre': metrics.JourneyMRE()}

    def fit(self, train_data_loader, valid_data_loader, num_epochs):
        history = dict(zip(['loss'] + list(self.metric_fns.keys()), [[] for _ in range(5)]))

        for epoch in range(1, 1 + num_epochs):
            # train
            self.network.train()
            batch_losses = 0.

            with tqdm.trange(len(train_data_loader)) as t:
                t.set_description('Epoch {} training'.format(epoch))
                for n, (inputs, targets, lengths) in zip(t, train_data_loader):
                    loss = self.train_step(inputs, targets, lengths)
                    batch_losses += loss
                    t.set_postfix(loss=loss)
                history['loss'].append(batch_losses / (n + 1))
            # evaluate
            results = self.evaluate(valid_data_loader)
            for k, v in results.items():
                history[k].append(v)

        return history

    def evaluate(self, data_loader):
        self.network.eval()

        for v in self.metric_fns.values():
            v.reset_states()
        with tqdm.trange(len(data_loader)) as t:
            t.set_description('Evaluation')
            for _, (inputs, targets, lengths) in zip(t, data_loader):
                results = self.eval_step(inputs, targets, lengths)
                t.set_postfix(**results)
        return {k: v.result().item() for (k, v) in self.metric_fns.items()}

    def train_step(self, inputs, targets, lengths):
        logits = self.network(inputs, lengths, targets)
        loss = self.loss_fn(logits, targets, lengths)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, inputs, targets, lengths):
        logits = self.network(inputs, lengths, None)
        evaluations = {}
        for k, v in self.metric_fns.items():
            evaluations.update({k: v(inputs, logits, targets, lengths).item()})
        return evaluations

    def _build_network(self):
        encoder_rnn = self.encoder_rnn_type(
            self.hidden_size, self.hidden_size, self.rnn_layers,
            dropout=self.encoder_rnn_dropout, bidirectional=False)
        encoder = networks.PointerEncoder(self.param_dim, encoder_rnn, dropout=self.encoder_dropout)
        decoder_rnn = self.decoder_rnn_type(
            self.hidden_size, self.hidden_size, self.rnn_layers,
            dropout=self.decoder_rnn_dropout, bidirectional=False)
        attention = self.attention_mechanism(self.hidden_size)
        decoder = networks.PointerDecoder(self.param_dim, decoder_rnn, attention, dropout=self.decoder_dropout)
        return networks.PointerNetwork(encoder, decoder)
