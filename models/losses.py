import torch


class TSPLoss(torch.nn.Module):
    def forward(self, inputs, targets, length=None):
        """TSP loss function.

        :param inputs: Padded output scores (logits) of shape (max_len - 1, batch_size, max_len).
        :param targets: Padded ground truth ranks of shape (max_len, batch_size), starting with index 0.
        :param length: Valid lengths of each example. Shape (batch_size,).
        :return:
        """
        del self
        max_len = targets.shape[0]
        mask = torch.where(torch.arange(max_len)[None, :] < length[:, None], torch.tensor(1.), torch.tensor(0.))
        loss = torch.nn.functional.cross_entropy(
            inputs.permute(1, 2, 0), targets[1:].permute(1, 0), reduction='none') * mask[:, 1:]
        loss = torch.mean(loss)
        return loss
