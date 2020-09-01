"""
logits: Pointer probabilities with insignificant padding, shape (max_len-1, batch_size, max_len).
targets: Predicted ranks with insignificant padding, shape (max_len, batch_size).
"""

import torch

from models import losses
from models import utils as model_utils


class Metric(torch.nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

    def reset_states(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def result(self):
        pass


class MeanMetric(Metric):
    def __init__(self, fn):
        super(MeanMetric, self).__init__()
        self.fn = fn
        self.total = torch.zeros(())
        self.count = torch.zeros(())

    def reset_states(self):
        self.total = torch.zeros(())
        self.count = torch.zeros(())

    def forward(self, *args, **kwargs):
        batch_metric_value = self.fn(*args, **kwargs)
        self.total += batch_metric_value.sum()
        self.count += batch_metric_value.size()[0]
        return self.result()

    def result(self):
        return self.total / self.count.to(dtype=self.total.dtype)


class TSPLoss(MeanMetric):
    def __init__(self):
        def metric_fn(_, logits, targets, lengths):
            return losses.batch_tsp_loss(logits, targets, lengths)

        super(TSPLoss, self).__init__(metric_fn)


def batch_bidirectional_accuracy(_, logits, targets, lengths):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor

    preds = logits.argmax(dim=-1).permute(1, 0)
    targets = targets.permute(1, 0)
    accuracies = []
    for o, t, l in zip(preds, targets, lengths):
        acc = max(
            torch.mean((o[: l] == t[:l]).to(dtype=logits.dtype)),
            torch.mean((o[: l].flip(0) == t[:l]).to(dtype=logits.dtype)))
        accuracies.append(acc)
    return torch.tensor(accuracies)


def _batch_journey_error(inputs, logits, targets, lengths):  # padded with zeros.
    preds = logits.argmax(dim=-1)
    max_len, batch_size = preds.shape

    pred_ranks = torch.cat([preds, torch.zeros(1, batch_size, dtype=preds.dtype)], dim=0)
    targets = torch.cat([targets, torch.zeros(1, batch_size, dtype=preds.dtype)], dim=0)
    pred = model_utils.permute_tensor(inputs, pred_ranks)
    pred = (pred[1:] - pred[:-1]).square().sum(dim=-1).sqrt().sum(dim=0)
    truth = model_utils.permute_tensor(inputs, targets)
    truth = (truth[1:] - truth[:-1]).square().sum(dim=-1).sqrt().sum(dim=0)
    return pred, truth


def batch_journey_mae(inputs, logits, targets, lengths):  # padded with zeros.
    pred, truth = _batch_journey_error(inputs, logits, targets, lengths)
    return pred - truth  # usually > 0


def batch_journey_mre(inputs, logits, targets, lengths):
    pred, truth = _batch_journey_error(inputs, logits, targets, lengths)
    return (pred - truth) / (truth + 1.e-6)


class BidirectionalAccuracy(MeanMetric):
    def __init__(self):
        super(BidirectionalAccuracy, self).__init__(batch_bidirectional_accuracy)


class JourneyMAE(MeanMetric):
    def __init__(self):
        super(JourneyMAE, self).__init__(batch_journey_mae)


class JourneyMRE(MeanMetric):
    def __init__(self):
        super(JourneyMRE, self).__init__(batch_journey_mre)
