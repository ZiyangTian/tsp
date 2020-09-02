import torch

from typing import Optional
from torch.nn.utils import rnn as rnn_utils


def batch_gather(sequence, indices):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """Batch gather from sequences.
        Arguments:
            sequence: A tensor of shape (batch_size, length, ...).
            indices: A tensor of shape (batch_size, num_indices).
        Returns:
            permuted_inputs: A tensor of shape (batch_size, num_indices, ...).
        """
    indices = torch.stack([indices] * sequence.shape[-1], dim=-1)
    permuted_inputs = torch.gather(sequence, 1, indices)
    return permuted_inputs


def batch_sequence_mask(lengths, max_len=None, dtype=torch.bool):
    # type: (torch.Tensor, Optional[int], Optional[torch.dtype]) -> torch.Tensor
    """Create a batch of sequence masks by valid lengths.
        Arguments:
            lengths: A `torch.int64` tensor of shape (batch_size,), the valid length of each example.
            max_len: An `int`, the maximum length to padding. Defaults to the maximum value of `lengths`.
            dtype: A `torch.dtype`, the output data type.
        Returns:
            mask: A binary tensor of shape (batch_size, max_len), true values for valid position.
    """
    return (torch.arange(max_len or lengths.max())[None, :] < lengths[:, None]).to(dtype=dtype)


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


def combine_masks(*masks, dtype=None):
    masks = torch.broadcast_tensors(*masks)
    mask = torch.ones_like(masks[0], dtype=torch.bool)
    for m in masks:
        mask = torch.logical_and(mask, m.to(dtype=torch.bool))
    return mask.to(dtype=dtype or masks[0].dtype)


def dynamic_rnn(rnn, inputs, lengths, hidden_state=None):
    packed_rnn_inputs = rnn_utils.pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
    packed_rnn_outputs, hidden_state = rnn(packed_rnn_inputs, hidden_state)
    padded_outputs, _ = rnn_utils.pad_packed_sequence(packed_rnn_outputs, batch_first=True)
    return padded_outputs, hidden_state
