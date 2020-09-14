import tqdm
import torch
import random

from models import losses
from models import metrics
from models import networks


class TSPModel(networks.PointerSequenceToSequence):
    def __init__(self, *args, **kwargs):
        super(TSPModel, self).__init__(*args, **kwargs)
        self.loss_fn = losses.TSPLoss()
        self.metric_fns = {
            # 'val_loss': metrics.TSPLoss(),
            'bidirectional_accuracy': metrics.BidirectionalAccuracy(),
            'journey_mae': metrics.JourneyMAE(),
            'journey_mre': metrics.JourneyMRE()}
        self.optimizer = None

    def compile(self, optimizer_obj, learning_rate=0.001, **kwargs):
        self.optimizer = optimizer_obj(self.parameters(), learning_rate, **kwargs)

    def fit(self, train_data_loader, eval_data_loader, num_epochs):
        history = dict(zip(['loss'] + list(self.metric_fns.keys()), [[] for _ in range(4)]))

        for epoch in range(1, 1 + num_epochs):
            epoch_loss = self.train_epoch(train_data_loader, description='Epoch {} training'.format(epoch))
            history['loss'].append(epoch_loss)
            evaluations = self.evaluate(eval_data_loader, description='Epoch {} evaluating'.format(epoch))
            for k, v in evaluations.items():
                history[k].append(v)
        return history

    def train_epoch(self, data_loader, description=None):
        self.train()
        batch_losses = 0.
        with tqdm.trange(len(data_loader)) as t:
            t.set_description(description)
            for n, data in zip(t, data_loader):
                if torch.rand(()) > 0.1:
                    continue
                loss = self.train_step(data)
                batch_losses += loss
                t.set_postfix(loss=loss)
            epoch_loss = batch_losses / (n + 1)
            t.set_postfix(loss=epoch_loss)
            return epoch_loss

    def evaluate(self, data_loader, description=None):
        self.eval()
        for v in self.metric_fns.values():
            v.reset_states()
        with tqdm.trange(len(data_loader)) as t:
            t.set_description(description)
            for _, data in zip(t, data_loader):
                predictions, evaluations = self.test_step(data)
                t.set_postfix(**evaluations)

        # plot the first
        # inputs, targets, lengths = data
        # i = random.randint(0, lengths.shape[0])
        # length = lengths[i]
        # parameters = inputs[i][:length]
        # target = targets[i][:length]
        # prediction = predictions[i][:length]
        # utils.plot_solution(parameters, target, prediction)

        return {k: v.result().item() for (k, v) in self.metric_fns.items()}

    def train_step(self, data):
        inputs, targets, lengths = data
        logits, _ = self(inputs, lengths=lengths, targets=targets, teacher_forcing_prob=1.)
        loss = self.loss_fn(logits, targets, lengths)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def test_step(self, data):
        inputs, targets, lengths = data
        _, predictions = self(inputs, lengths=lengths, targets=None, teacher_forcing_prob=0.)
        evaluations = {}
        for k, v in self.metric_fns.items():
            evaluations.update({k: v(inputs, predictions, targets, lengths).item()})
        return predictions, evaluations
