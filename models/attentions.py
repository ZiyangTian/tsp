"""Attention mechanism."""
import torch

from typing import Callable, Optional


class Attention(torch.nn.Module):
    """Base class for attention mechanisms.
        Arguments:
            score_fn: A score function to compute score tensor from query and key.
                Follow the signature: (torch.Tensor, torch.Tensor) -> torch.Tensor.
    """
    def __init__(self, score_fn):
        # type: (Attention, Callable) -> None
        super(Attention, self).__init__()
        self.score_fn = score_fn

    def forward(self, query, key, value=None):
        # type: (Attention, torch.Tensor, torch.Tensor, Optional[None, torch.Tensor]) -> (torch.Tensor, torch.Tensor)
        """Forward propagation function.
        Arguments:
            query: Query tensor of shape (..., query_length, query_hidden_size).
            key: Key tensor of shape (..., kv_length, key_hidden_size).
            value (optional): Value tensor of shape (..., kv_length, value_hidden_size).
            `query`, `key` and `value` must have matched leading dimensions.
        Returns:
            score: Attention score of shape (..., query_length, kv_length).
            context_vector (optional): Context vector of shape (..., query_length, value_hidden_size).
                Only returns when `value` is specified.
        """
        score = self.score_fn(query, key)  # (..., query_length, kv_length)
        if value is None:
            return score

        attention_weights = score.softmax(dim=-1)
        context_vector = torch.bmm(attention_weights, value)
        return attention_weights, context_vector


class BahdanauAttention(Attention):
    """Bahdanau attention mechanism.
        Arguments:
            query_hidden_size: Hidden size, representing the identity qk_hidden_size and value_hidden_size.
    """
    def __init__(self, hidden_size, query_hidden_size=None, key_hidden_size=None, value_hidden_size=None):
        # type: (BahdanauAttention, int, Optional[None, int], Optional[None, int], Optional[None, int]) -> None
        def score_fn(query, key):
            query = query[..., :, None, :]
            key = key[..., None, :, :]
            return self.v(torch.tanh(self.w1(query) + self.w2(key))).squeeze(dim=-1)

        super(BahdanauAttention, self).__init__(score_fn)
        self.hidden_size = hidden_size
        self.query_hidden_size = query_hidden_size or hidden_size
        self.key_hidden_size = key_hidden_size or hidden_size
        self.value_hidden_size = value_hidden_size or hidden_size

        self.w1 = torch.nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.w2 = torch.nn.Linear(self.key_hidden_size, self.hidden_size, bias=False)
        self.v = torch.nn.Linear(self.hidden_size, 1, bias=False)


class LoungAttention(Attention):
    """Loung attention mechanism.
        Arguments:
            hidden_size: Hidden size, representing the identity qk_hidden_size and value_hidden_size.
    """
    def __init__(self, hidden_size):
        # type: (LoungAttention, int) -> None
        def score_fn(query, key):
            return self.w(query).bmm(key.permute(0, 2, 1))

        super(LoungAttention, self).__init__(score_fn)
        self.hidden_size = hidden_size

        self.w = torch.nn.Linear(hidden_size, hidden_size, bias=False)
