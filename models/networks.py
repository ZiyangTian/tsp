"""Pointer networks."""
import functools
import torch

from typing import Any, Dict, Optional, Type
from torch.nn.utils import rnn as rnn_utils

from models import attentions
from models import utils


class PointerEncoder(torch.nn.Module):
    """Encoder of a pointer network.
        Arguments:
            input_size: Dimension of the inputs.
            rnn: RNN module.
            dropout: Dropout probability.
    """
    def __init__(self, input_size, rnn, dropout=0.):
        # type: (PointerEncoder, int, torch.nn.RNNBase, float) -> None
        super(PointerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = rnn.hidden_size
        self.num_layers = rnn.num_layers

        self.linear = torch.nn.Linear(input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.rnn = rnn

    def forward(self, inputs, lengths):
        # type: (PointerEncoder, torch.Tensor, torch.Tensor) -> (torch.Tensor, torch.Tensor)
        """Forward propagation function.
        Arguments:
            inputs: Padded encoder inputs of shape (max_length, batch_size, input_size).
            lengths: Sequence lengths of each example. Shape: (batch_size,)
        Returns:
            padded_rnn_outputs: Padded encoder outputs of shape (max_length, batch_size, hidden_size).
            hidden: Hidden states of shape (num_rnn_layers, batch_size, hidden_size).
        """
        rnn_inputs = self.linear(inputs)
        rnn_inputs = self.dropout(rnn_inputs)
        packed_rnn_inputs = rnn_utils.pack_padded_sequence(rnn_inputs, lengths, enforce_sorted=False)
        packed_rnn_outputs, hidden = self.rnn(packed_rnn_inputs)
        padded_outputs, _ = rnn_utils.pad_packed_sequence(packed_rnn_outputs)
        return padded_outputs, hidden


class PointerDecoder(torch.nn.Module):
    """Decoder of a pointer network.
        Arguments:
            input_size: Dimension of the inputs.
            rnn: RNN module.
            attention: Attention object.
            dropout: Dropout probability.
    """
    def __init__(self, input_size, rnn, attention, dropout=0.):
        # type: (PointerDecoder, int, torch.nn.RNNBase, attentions.Attention, float) -> None
        super(PointerDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = rnn.hidden_size
        self.num_layers = rnn.num_layers

        self.linear = torch.nn.Linear(input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = functools.partial(attention, value=None)
        self.rnn = rnn

    def forward(self, inputs, encoder_outputs, encoder_hidden, target_ranks=None):
        # type: (PointerDecoder, torch.Tensor, torch.Tensor, torch.Tensor, Optional[None, torch.Tensor]) -> (torch.Tensor, torch.Tensor)
        """Forward propagation function.
        Arguments:
            inputs: Padded inputs of shape (max_len, batch_size, input_size). Must be identity to
                the encoder input.
            encoder_outputs: Encoder output of shape (max_len, batch_size, hidden_size)
            encoder_hidden: Encoder hidden state of shape (num_encoder_rnn_layers, batch_size, hidden_size)
            target_ranks (optional): Ground truth of output ranks. Shape (max_len, batch_size).
                If specified, use teacher forcing.
        Returns:
            output_scores: Pointer probabilities with insignificant padding, shape (max_len-1, batch_size, max_len).
            ranks: Predicted ranks with insignificant padding, shape (max_len, batch_size).
        """
        rnn_inputs = self.linear(inputs)
        max_len, batch_size, _ = rnn_inputs.shape
        initial_input_index = torch.zeros(1, batch_size, dtype=torch.int64)

        if target_ranks is not None:
            decoder_inputs = utils.permute_inputs(rnn_inputs, target_ranks)
            decoder_outputs, _ = self.rnn(decoder_inputs, encoder_hidden)
            output_scores = self.attention(
                decoder_outputs.permute(1, 0, 2),
                encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)[:-1]  # (max_len-1, batch_size, max_len)
        else:
            hidden = encoder_hidden
            input_index = initial_input_index
            attention_scores = []
            for i in range(max_len - 1):
                rnn_input = utils.permute_inputs(rnn_inputs, input_index)  # (1, batch_size, hidden_size)
                rnn_output, hidden = self.rnn(rnn_input, hidden)
                attention_score = self.attention(
                    rnn_output.permute(1, 0, 2),
                    encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
                input_index = attention_score.argmax(dim=-1)  # (batch_size)
                attention_scores.append(attention_score)
            output_scores = torch.cat(attention_scores, dim=0)  # (max_len-1, batch_size, max_len)
        return output_scores


class PointerNetwork(torch.nn.Module):
    """Pointer network.
        Arguments:
            encoder: Encoder of the pointer network.
            decoder: Decoder of the pointer network.
        Raises:
            ValueError: If the RNN parameters of the encoder and the decoder are incompatible.
    """
    def __init__(self, encoder, decoder):
        # type: (PointerNetwork, PointerEncoder, PointerDecoder) -> None
        super(PointerNetwork, self).__init__()
        self.input_size = encoder.input_size
        self.hidden_size = encoder.hidden_size
        self.num_layers = encoder.num_layers
        if self.hidden_size != decoder.hidden_size or self.num_layers != decoder.num_layers:
            raise ValueError('Incompatible RNN parameters of the encoder and the decoder.')

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, lengths, target_ranks=None):
        # type: (PointerNetwork, torch.Tensor, torch.Tensor, Optional[None, torch.Tensor]) -> (torch.Tensor, torch.Tensor)
        """Forward propagation function.
        Arguments:
            inputs: Padded inputs of shape (max_len, batch_size, input_size).
            lengths: Sequence lengths of each example. Shape: (batch_size,).
            target_ranks: (optional): Ground truth of output ranks. Shape (max_len, batch_size).
                If specified, use teacher forcing.
        Returns:
            output_scores: Pointer probabilities with insignificant padding, shape (max_len, batch_size, max_len).
        """
        max_len, batch_size, _ = inputs.size()
        encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)
        output_scores = self.decoder(inputs, encoder_outputs, encoder_hidden, target_ranks=target_ranks)
        first_logit = torch.nn.functional.one_hot(  # Manually adding the prediction for the first step.
            torch.zeros(batch_size, dtype=torch.int64),
            num_classes=max_len).to(dtype=output_scores.dtype)[None, ...] * 10e9
        logits = torch.cat([first_logit, output_scores], dim=0)
        return logits

    @classmethod
    def from_config(cls, config):
        # type: (Type, Dict[str, Any]) -> PointerNetwork
        input_size = config.pop('input_size')
        hidden_size = config.pop('hidden_size')
        rnn_layers = config.pop('rnn_layers', 1)
        encoder_dropout = config.pop('encoder_dropout', 0.)
        encoder_rnn_type = config.pop('encoder_rnn_type', torch.nn.LSTM)
        encoder_rnn_dropout = config.pop('encoder_rnn_dropout', 0.)
        decoder_dropout = config.pop('decoder_dropout', 0.)
        decoder_rnn_type = config.pop('decoder_rnn_type', torch.nn.LSTM)
        decoder_rnn_dropout = config.pop('decoder_rnn_dropout', 0.)
        attention_mechanism = config.pop('attention_mechanism', attentions.BahdanauAttention)

        encoder_rnn = encoder_rnn_type(
            hidden_size, hidden_size, rnn_layers, dropout=encoder_rnn_dropout, bidirectional=False)
        encoder = PointerEncoder(input_size, encoder_rnn, dropout=encoder_dropout)
        decoder_rnn = decoder_rnn_type(
            hidden_size, hidden_size, rnn_layers, dropout=decoder_rnn_dropout, bidirectional=False)
        attention = attention_mechanism(hidden_size)
        decoder = PointerDecoder(input_size, decoder_rnn, attention, dropout=decoder_dropout)
        model = cls(encoder, decoder)
        return model
