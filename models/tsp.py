import tqdm
import torch

from models import attentions
from models import losses
from models import metrics
from models import networks


class TSPModel(object):
    param_dim = 2

    hidden_size = 128
    rnn_layers = 1
    rnn_type = torch.nn.LSTM
    rnn_bidirectional = False
    input_dropout = 0.
    encoder_rnn_dropout = 0.
    decoder_rnn_dropout = 0.
    attention_mechanism = attentions.BahdanauAttention
    attention_hidden_size = None

    optimizer_obj = torch.optim.Adam
    learning_rate = 0.001
    optimizer_kwargs = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.network = networks.PointerNetwork(
            input_size=self.param_dim,
            hidden_size=self.hidden_size,
            rnn_type=self.rnn_type,
            rnn_layers=self.rnn_layers,
            rnn_bidirectional=self.rnn_bidirectional,
            input_dropout=self.input_dropout,
            encoder_rnn_dropout=self.encoder_rnn_dropout,
            decoder_rnn_dropout=self.decoder_rnn_dropout,
            attention_mechanism=self.attention_mechanism,
            attention_hidden_size=self.attention_hidden_size)
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
        logits = self.network(inputs, lengths=lengths, targets=targets)
        loss = self.loss_fn(logits, targets, lengths)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def eval_step(self, inputs, targets, lengths):
        logits = self.network(inputs, lengths, None)
        evaluations = {}
        for k, v in self.metric_fns.items():
            evaluations.update({k: v(inputs, logits, targets, lengths).item()})
        return evaluations
