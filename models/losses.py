import torch

from models import utils as model_utils


def batch_tsp_loss(logits, targets, lengths):
    # logits: (max_len, batch_size, max_len)
    # targets: (max_len, batch_size)
    # lengths: (batch_size, )
    mask = model_utils.batch_sequence_mask(lengths, logits.size()[1], dtype=logits.dtype)
    loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1), targets, reduction='none') * mask
    loss = loss.mean(dim=-1)
    return loss


class TSPLoss(torch.nn.Module):
    @staticmethod
    def forward(logits, targets, lengths):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """TSP loss function.
        Arguments:
            logits: Padded logits of shape (max_len, batch_size, max_len).
            targets: Padded ground truth ranks of shape (max_len, batch_size), starting with index 0.
            lengths: Valid lengths of each example. Shape (batch_size,).
        Returns:
            loss: Scalar loss value.
        """
        batch_loss = batch_tsp_loss(logits, targets, lengths)
        return batch_loss.mean()
