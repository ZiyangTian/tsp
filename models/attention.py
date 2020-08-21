"""Attention mechanism."""
import torch

from typing import Callable


class Attention(torch.nn.Module):
    """Base class for attention mechanisms.
        Arguments:
            score_fn: A score function to compute score tensor from query, key and value.
                Follow the signature: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor.
                See `forward`.
    """
    def __init__(self, score_fn):
        # type: (Attention, Callable) -> None
        super(Attention, self).__init__()
        self.score_fn = score_fn

    def forward(self, query, key, value):
        # type: (Attention, torch.Tensor, torch.Tensor, torch.Tensor) -> (torch.Tensor, torch.Tensor)
        """Forward propagation function.
        Arguments:
            query: Query tensor of shape (..., query_length, qk_hidden_size).
            key: Key tensor of shape (..., kv_length, qk_hidden_size).
            value: Value tensor of shape (..., kv_length, value_hidden_size).
            `query`, `key` and `value` must have matched leading dimensions.
        Returns:
            attention_weights: Attention weights of shape (..., query_length, kv_length).
            context_vector: Context vector of shape (..., query_length, value_hidden_size).
        """
        score = self.score_fn(query, key, value)  # (..., query_length, kv_length)
        attention_weights = torch.nn.functional.softmax(score, dim=-1)
        context_vector = torch.bmm(attention_weights, value)
        return attention_weights, context_vector


class BahdanauAttention(Attention):
    """Bahdanau attention mechanism.
        Arguments:
            hidden_size: Hidden size, representing the identity qk_hidden_size and value_hidden_size.
    """
    def __init__(self, hidden_size):
        # type: (BahdanauAttention, int) -> None
        self.hidden_size = hidden_size

        self.w1 = torch.nn.Linear(hidden_size, hidden_size)
        self.w2 = torch.nn.Linear(hidden_size, hidden_size)
        self.v = torch.nn.Linear(hidden_size, hidden_size)

        def score_fn(query, key, value):
            del key
            return self.v(torch.tanh(self.w1(query) + self.w2(value)))

        super(BahdanauAttention, self).__init__(score_fn)


class LoungAttention(Attention):
    """Loung attention mechanism.
        Arguments:
            hidden_size: Hidden size, representing the identity qk_hidden_size and value_hidden_size.
    """
    def __init__(self, hidden_size):
        # type: (LoungAttention, int) -> None
        self.hidden_size = hidden_size

        self.w = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        def score_fn(query, key, value):
            del key
            return self.w(query).bmm(value.permute(0, 2, 1))

        super(LoungAttention, self).__init__(score_fn)
