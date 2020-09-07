"""
logits: Pointer probabilities with insignificant padding, shape (max_len-1, batch_size, max_len).
targets: Predicted ranks with insignificant padding, shape (max_len, batch_size).
"""
import abc
import torch

from models import losses
from models import utils as model_utils


class Metric(torch.nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.eval()

    @abc.abstractmethod
    def reset_states(self):
        raise NotImplementedError('Metric.reset_states')

    @abc.abstractmethod
    def result(self):
        raise NotImplementedError('Metric.result')


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
        with torch.no_grad():
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


def _accuracy(predictions, targets, mask):
    accuracies = (predictions == targets).to(dtype=torch.float32) * mask.to(dtype=torch.float32)
    return accuracies.sum(-1) / mask.sum(-1)


def batch_bidirectional_accuracy(inputs, predictions, targets, lengths):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    del inputs
    reversed_targets = model_utils.batch_reverse_sequence(targets, lengths - 1)
    mask = model_utils.batch_sequence_mask(lengths, max_len=targets.shape[-1], dtype=torch.float32)
    acc_1 = _accuracy(predictions, targets, mask)
    acc_2 = _accuracy(predictions, reversed_targets, mask)
    return torch.where(acc_1 > acc_2, acc_1, acc_2)


def _batch_journey_error(inputs, predictions, targets, lengths):  # padded with zeros.
    pred_journey = model_utils.batch_journey(inputs, predictions, lengths)
    truth_journey = model_utils.batch_journey(inputs, targets, lengths)
    return pred_journey, truth_journey


def batch_journey_mae(inputs, predictions, targets, lengths):  # padded with zeros.
    pred_journey, truth_journey = _batch_journey_error(inputs, predictions, targets, lengths)
    return pred_journey - truth_journey  # usually > 0


def batch_journey_mre(inputs, predictions, targets, lengths):
    pred_journey, truth_journey = _batch_journey_error(inputs, predictions, targets, lengths)
    return (pred_journey - truth_journey) / (truth_journey + 1.e-6)


class BidirectionalAccuracy(MeanMetric):
    def __init__(self):
        super(BidirectionalAccuracy, self).__init__(batch_bidirectional_accuracy)


class JourneyMAE(MeanMetric):
    def __init__(self):
        super(JourneyMAE, self).__init__(batch_journey_mae)


class JourneyMRE(MeanMetric):
    def __init__(self):
        super(JourneyMRE, self).__init__(batch_journey_mre)
