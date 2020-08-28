import torch

from typing import Optional


def permute_inputs(inputs, ranks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """Forward propagation function.
        Arguments:
            inputs: Padded inputs of shape (input_length, batch_size, input_size). Must be identity to
                the encoder input.
            ranks (optional): Ground truth of output ranks. Shape (target_length, batch_size).
                If specified, use teacher forcing.
        Returns:
            permuted_inputs: Permuted inputs of shape (target_length, batch_size, input_size).
        """
    ranks = torch.stack([ranks] * inputs.shape[-1], dim=-1)
    permuted_inputs = torch.gather(inputs, 0, ranks)
    return permuted_inputs


def batch_sequence_mask(lengths, max_len=None, dtype=None):
    # type: (torch.Tensor, Optional[int], Optional[torch.dtype]) -> torch.Tensor
    """Mask a batch of sequence by valid lengths.
        Arguments:
            lengths: The valid length of each example. Shape (batch_size,).
            max_len: The maximum length to padding.
            dtype: The output data type, defaults to `torch.int64`.
        Returns:
            mask: A binary tensor, true values for valid.
    """
    if max_len is None:
        max_len = lengths.max()
    return torch.where(
        torch.arange(max_len)[None, :] < lengths[:, None],
        torch.tensor(1),
        torch.tensor(0)).to(dtype=dtype)


if __name__ == '__main__':
    a = torch.arange(24).reshape(4, 3, 2)
    b = torch.tensor([[0, 3, 1],
                      [0, 2, 1],
                      [0, 1, 0]])
    c = permute_inputs(a, b)
    print(a.permute(1, 0, 2), c.permute(1, 0, 2))

    # d = torch.tensor([0, 3, 1])
    # e = torch.index_select(a, 0, d)
