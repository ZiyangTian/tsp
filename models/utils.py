import torch


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


if __name__ == '__main__':
    a = torch.arange(24).reshape(4, 3, 2)
    b = torch.tensor([[0, 3, 1],
                      [0, 2, 1],
                      [0, 1, 0]])
    c = permute_inputs(a, b)
    print(a.permute(1, 0, 2), c.permute(1, 0, 2))

    # d = torch.tensor([0, 3, 1])
    # e = torch.index_select(a, 0, d)
