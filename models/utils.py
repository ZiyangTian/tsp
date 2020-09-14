import numpy as np
import torch
import time
import matplotlib.pyplot as plt

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


def batch_reverse_sequence(sequences, lengths):
    indices = torch.arange(sequences.shape[-1], dtype=torch.int64)
    indices, lengths = torch.broadcast_tensors(indices[None, :], lengths[:, None])
    gather_indices = torch.where(
        indices < lengths,
        lengths - indices - 1,
        indices)
    return torch.gather(sequences, 1, gather_indices)


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


def batch_journey(parameters, ending, lengths):
    # indices: starting zero at end
    mask = batch_sequence_mask(lengths, max_len=ending.shape[-1], dtype=parameters.dtype)
    ending = batch_gather(parameters, ending)
    beginning = ending.roll(1, dims=1)
    return ((ending - beginning).square().sum(-1) * mask).sqrt().sum(-1)


def combine_masks(*masks, dtype=None):
    masks = torch.broadcast_tensors(*masks)
    mask = torch.ones_like(masks[0], dtype=torch.bool)
    for m in masks:
        mask = torch.logical_and(mask, m.to(dtype=torch.bool))
    return mask.to(dtype=dtype or masks[0].dtype)


def dynamic_rnn(rnn, inputs, lengths=None, hidden_state=None):
    if lengths is None:
        return rnn(inputs, hidden_state)
    packed_rnn_inputs = rnn_utils.pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
    packed_rnn_outputs, hidden_state = rnn(packed_rnn_inputs, hidden_state)
    padded_outputs, _ = rnn_utils.pad_packed_sequence(packed_rnn_outputs, batch_first=True)
    return padded_outputs, hidden_state


def stack_n(tensor, n, dim):
    return torch.stack([tensor] * n, dim=dim)


def _get_numpy(maybe_tensor):
    if isinstance(maybe_tensor, torch.Tensor):
        if maybe_tensor.requires_grad:
            maybe_tensor = maybe_tensor.detach()
        return maybe_tensor.cpu().numpy()
    return np.array(maybe_tensor)


def plot_solution(parameters, target, prediction):
    parameters, target, prediction = _get_numpy(parameters), _get_numpy(target), _get_numpy(prediction)
    target = np.concatenate([target, target[0:1]], axis=0)
    prediction = np.concatenate([prediction, prediction[0:1]], axis=0)

    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(parameters[:, 0][target], parameters[:, 1][target], 'r-', color='green')
    plt.plot(parameters[:, 0][prediction], parameters[:, 1][prediction], 'r--', color='blue')
    plt.ion()
    plt.pause(1)
    plt.close()
    # plt.show()
