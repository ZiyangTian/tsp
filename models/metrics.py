"""
output_scores: Pointer probabilities with insignificant padding, shape (max_len-1, batch_size, max_len).
ranks: Predicted ranks with insignificant padding, shape (max_len, batch_size).
"""

import torch

from models import utils as model_utils


class Metric(torch.nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

    def reset_states(self):
        pass

    def update_state(self, *args, **kwargs):
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

    def update_state(self, *args, **kwargs):
        self.total += self.fn(*args, **kwargs)
        self.count += 1.

    def result(self):
        return self.total / self.count


def accuracy(output, target, weights=None):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    true_pred = (output == target).to(dtype=output.dtype)
    if weights is None:
        return torch.mean(true_pred)
    return torch.sum(true_pred * weights) / torch.mean(weights)


def bidirectional_accuracy(pred_ranks, target_ranks, lengths):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    pred_ranks = pred_ranks.permute(1, 0)
    target_ranks = target_ranks.permute(1, 0)
    accuracies = []
    for o, t, l in zip(pred_ranks, target_ranks, lengths):
        acc = max(
            torch.mean((o[: lengths] == t[:lengths]).to(dtype=pred_ranks.dtype)),
            torch.mean((o[: lengths].flip(0) == t[:lengths]).to(dtype=pred_ranks.dtype)))
        accuracies.append(acc)
    return torch.mean(torch.tensor(accuracies))


def journey_mae(parameters, outputs, targets, lengths):
    max_len, batch_size = outputs.shape

    outputs = torch.cat([outputs, torch.zeros(1, batch_size, dtype=torch.int64)], dim=0)
    targets = torch.cat([targets, torch.zeros(1, batch_size, dtype=torch.int64)], dim=0)
    lengths = lengths + 1
    pred = model_utils.permute_inputs(parameters, outputs)
    pred =
    truth = model_utils.permute_inputs(parameters, targets)

class Accuracy(MeanMetric):
    def __init__(self):
        super(Accuracy, self).__init__(bidirectional_accuracy)
