"""Attention mechanism."""
import abc
import torch

from typing import Optional


class Attention(torch.nn.Module):
    """Base class for attention mechanisms."""
    def forward(self, query, key, value=None, mask=None):
        """Forward propagation function.
        Arguments:
            query: Query tensor of shape (batch_size, query_length, query_hidden_size).
            key: Key tensor of shape (batch_size, kv_length, key_hidden_size).
            value (optional): Value tensor of shape (batch_size, kv_length, value_hidden_size).
            mask (optional): Mask tensor of shape (batch_size, query_length, kv_length).
        Returns:
            score: Attention score of shape (batch_size, query_length, kv_length).
            context_vector (optional): Context vector of shape (batch_size, query_length, value_hidden_size).
                Only returns when `value` is specified.
        """
        score = self._score_fn(query, key)  # (batch_size, query_length, kv_length)
        score += - 1e-9 * (1 - mask.to(dtype=query.dtype))
        if value is None:
            return score

        attention_weights = score.softmax(dim=-1)
        context_vector = attention_weights.bmm(value.permute(0, 2, 1))
        return attention_weights, context_vector.permute

    @abc.abstractmethod
    def _score_fn(self, query, key):
        """A score function to compute score tensor from query and key.
        Arguments:
            query: Query tensor of shape (batch_size, query_length, query_hidden_size).
            key: Key tensor of shape (batch_size, kv_length, query_hidden_size).
        Returns:
            score: Attention score of shape (batch_size, query_length, kv_length).
        """
        raise NotImplementedError('Attention._score_fn')


class BahdanauAttention(Attention):
    """Bahdanau attention mechanism.
        Arguments:
            hidden_size: An `int`, attention hidden_state size.
            query_hidden_size: An `int`, query hidden_state size. Defaults to be identity to `hidden_size`.
            key_hidden_size: An `int`, key hidden_state size. Defaults to be identity to `hidden_size`.
    """
    def __init__(self, hidden_size, query_hidden_size=None, key_hidden_size=None):
        # type: (BahdanauAttention, int, Optional[None, int], Optional[None, int]) -> None
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_hidden_size = query_hidden_size or hidden_size
        self.key_hidden_size = key_hidden_size or hidden_size

        self.wq = torch.nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.wk = torch.nn.Linear(self.key_hidden_size, self.hidden_size, bias=False)
        self.v = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def _score_fn(self, query, key):
        query = query.unsqueeze(-2)
        key = key.unsqueeze(-3)
        return self.v(torch.tanh(self.wq(query) + self.wk(key))).squeeze(dim=-1)


class LoungAttention(Attention):
    """Loung attention mechanism.
        Arguments:
            hidden_size: An `int`, attention hidden_state size.
            query_hidden_size: An `int`, query hidden_state size. Defaults to be identity to `hidden_size`.
            key_hidden_size: An `int`, key hidden_state size. Defaults to be identity to `hidden_size`.
    """
    def __init__(self, hidden_size, query_hidden_size=None, key_hidden_size=None):
        # type: (LoungAttention, int, Optional[None, int], Optional[None, int]) -> None
        super(LoungAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_hidden_size = query_hidden_size or hidden_size
        self.key_hidden_size = key_hidden_size or hidden_size

        self.wq = torch.nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.wk = torch.nn.Linear(self.key_hidden_size, self.hidden_size, bias=False)

    def _score_fn(self, query, key):
        query = self.wq(query)
        key = self.wk(key)
        return query.bmm(key.permute(0, 2, 1))
