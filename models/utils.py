import torch

from typing import Optional
from torch.nn.utils import rnn as rnn_utils


def permute_tensor(sequence, ranks):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """Forward propagation function.
        Arguments:
            sequence: Padded tensor of shape (batch_size, input_length, input_size). Must be identity to
                the encoder input.
            ranks (optional): Ground truth of output ranks. Shape (batch_size, target_length).
                If specified, use teacher forcing.
        Returns:
            permuted_inputs: Permuted inputs of shape (batch_size, target_length, input_size).
        """
    ranks = torch.stack([ranks] * sequence.shape[-1], dim=-1)
    permuted_inputs = torch.gather(sequence, 1, ranks)
    return permuted_inputs


def batch_sequence_mask(lengths, max_len=None, dtype=torch.bool):
    # type: (torch.Tensor, Optional[int], Optional[torch.dtype]) -> torch.Tensor
    """Mask a batch of sequence by valid lengths.
        Arguments:
            lengths: The valid length of each example. Shape (batch_size,).
            max_len: The maximum length to padding.
            dtype: The output data type, defaults to `torch.bool`.
        Returns:
            mask: A binary tensor, true values for valid. Shape (batch_size, max_len).
    """
    if max_len is None:
        max_len = lengths.max()
    return (torch.arange(max_len)[None, :] < lengths[:, None]).to(dtype=dtype)


# def parallel_batch_permutation_mask(permutations):
#     # (max_len, batch_size) must be a full permutation
#     max_len, batch_size = permutations.size()
#     indices = torch.arange(max_len)
#     # a = torch.where(indices[:, None, None] - indices_last >= 1, permutations.unsqueeze(-1), indices_last)
#     permutations, bools = torch.broadcast_tensors(
#         permutations.permute(1, 0).unsqueeze(0),
#         indices[:, None, None] - indices[None, None, :] <= 0)
#     mask = torch.ones(max_len, batch_size, max_len, dtype=torch.bool).scatter(-1, permutations, bools)
#     return mask


def batch_step_mask(masked, decoded):
    """Create a new mask by adding a decoded position to an old mask.
        Arguments:
            masked: The old mask of shape (batch_size, max_len).
            decoded: The decoded positions of shape (batch_size,).
        Returns:
            A new mask of shape (batch_size, max_len).
    """
    decoded = decoded.unsqueeze(-1)
    return masked.scatter(-1, decoded, torch.zeros_like(decoded, dtype=masked.dtype))


def batch_steps_mask(steps, dtype=torch.bool):
    """Create a step mask by masking decoded position at each step.
        Arguments:
            steps: The index sequence of shape (batch_size, max_len).
            dtype: Output data type.
        Returns:
            A mask of shape (batch_size, max_len, max_len).
    """
    batch_size, max_len = steps.size()
    masked = torch.ones(batch_size, max_len, dtype=dtype)
    masks = [masked]
    for i in range(max_len - 1):
        masked = batch_step_mask(masked, steps[:, i])
        masks.append(masked)
    return torch.stack(masks, dim=1)


# def attention_mask(targets=None, lengths=None):
#     padding_mask = utils.batch_sequence_mask(lengths)  # (batch_size, max_len)
#     steps_mask = utils.batch_steps_mask(targets)  # (batch_size, max_len, max_len)
#     attention_mask = utils.combine_masks(
#         padding_mask.unsqueeze(dim=-1),
#         padding_mask.unsqueeze(dim=-2),
#         steps_mask)


def combine_masks(*masks, dtype=None):
    masks = torch.broadcast_tensors(*masks)
    mask = torch.ones_like(masks[0], dtype=torch.bool)
    for m in masks:
        mask = torch.logical_and(mask, m.to(dtype=torch.bool))
    return mask.to(dtype=dtype or masks[0].dtype)


def dynamic_rnn(rnn, inputs, lengths, hidden_state=None):
    print(inputs.shape, lengths.shape)
    packed_rnn_inputs = rnn_utils.pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
    packed_rnn_outputs, hidden_state = rnn(packed_rnn_inputs, hidden_state)
    padded_outputs, _ = rnn_utils.pad_packed_sequence(packed_rnn_outputs, batch_first=True)
    return padded_outputs, hidden_state


def main():
    permutations = torch.tensor([
        [0, 3, 0, 4, 1],
        [0, 2, 1, 0, 0],
        [0, 1, 4, 2, 3]])
    mask = batch_steps_mask(permutations, dtype=torch.int)
    print(mask)

    # d = torch.tensor([0, 3, 1])
    # e = torch.index_select(a, 0, d)
    # masked = torch.tensor([
    #          [0, 1, 1, 0, 1],
    #          [0, 1, 0, 1, 1],
    #          [0, 0, 1, 1, 1]], dtype=torch.int32)
    # decoded = torch.tensor([2, 3, 4])
    # new_mask = batch_permutation_decoding_mask(masked, decoded)
    # print(new_mask)


if __name__ == '__main__':
    main()
