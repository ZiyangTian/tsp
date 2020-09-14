"""Attention mechanism."""
import abc
import torch


class Attention(torch.nn.Module):
    """Base class for attention mechanisms."""
    def __init__(self, hidden_size, use_scale=False, dropout=0.):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        if use_scale:
            self.scale = torch.tensor(self.hidden_size, dtype=torch.float32).sqrt().reciprocal()
        else:
            self.scale = torch.ones((), dtype=torch.float32)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value=None, mask=None):
        """Forward propagation function.
        Arguments:
            query: A tensor of shape (batch_size, query_length, query_hidden_size).
            key: A tensor of shape (batch_size, kv_length, key_hidden_size).
            value (optional): A tensor of shape (batch_size, kv_length, hidden_size).
            mask (optional): A tensor of shape (batch_size, query_length, kv_length).
        Returns:
            score: Attention score of shape (batch_size, query_length, kv_length).
            context_vector (optional): Context vector of shape (batch_size, query_length, hidden_size).
                Only returns when `value` is specified.
        """
        score = self._score_fn(query, key) * self.scale  # (batch_size, query_length, kv_length)
        score += -1e9 * (1 - mask.to(dtype=query.dtype))
        if value is None:
            return score

        attention_weights = self.dropout(score.softmax(dim=-1))
        context_vector = attention_weights.bmm(value)
        return context_vector, attention_weights

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
    def __init__(self, hidden_size, query_hidden_size=None, key_hidden_size=None, use_scale=False, dropout=0.):
        super(BahdanauAttention, self).__init__(hidden_size, use_scale=use_scale, dropout=dropout)
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
    def __init__(self, hidden_size, query_hidden_size=None, key_hidden_size=None, use_scale=False, dropout=0.):
        super(LoungAttention, self).__init__(hidden_size, use_scale=use_scale, dropout=dropout)
        self.query_hidden_size = query_hidden_size or hidden_size
        self.key_hidden_size = key_hidden_size or hidden_size

        self.wq = torch.nn.Linear(self.query_hidden_size, self.hidden_size, bias=False)
        self.wk = torch.nn.Linear(self.key_hidden_size, self.hidden_size, bias=False)

    def _score_fn(self, query, key):
        query = self.wq(query)
        key = self.wk(key)
        return query.bmm(key.permute(0, 2, 1))


class DotProductAttention(Attention):
    def _score_fn(self, query, key):
        return query.bmm(key.permute(0, 2, 1))
