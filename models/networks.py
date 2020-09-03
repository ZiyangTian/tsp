"""Pointer networks."""
import torch

from models import attentions
from models import utils


class PointerNetwork(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type=torch.nn.GRU,
                 rnn_layers=1,
                 input_dropout=0.,
                 encoder_rnn_dropout=0.,
                 decoder_rnn_dropout=0.,
                 attention_mechanism=attentions.BahdanauAttention,
                 attention_hidden_size=None):
        super(PointerNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = torch.nn.Linear(self.input_size, self.hidden_size)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.encoder_rnn = rnn_type(
            self.hidden_size,
            self.hidden_size,
            num_layers=rnn_layers,
            dropout=encoder_rnn_dropout,
            batch_first=True)
        self.decoder_rnn = rnn_type(
            self.hidden_size,
            self.hidden_size,
            num_layers=rnn_layers,
            dropout=decoder_rnn_dropout,
            batch_first=True)
        self.attention = attention_mechanism(
            attention_hidden_size or self.hidden_size,
            query_hidden_size=self.hidden_size,
            key_hidden_size=self.hidden_size)

    def forward(self, inputs, targets=None, lengths=None):
        """Forward propagation function.
        Arguments:
            inputs: A `torch.float32` tensor of shape (batch_size, max_len, input_size), representing
                the input parameters.
            targets: A `torch.int64` tensor of shape (batch_size, max_len), representing the ground
                truth of output ranks. If specified, use teacher forcing.
            lengths: A `torch.int64` tensor of shape (batch_size,), representing the valid sequence
                lengths of each example.
        Returns:
            logits: A `torch.float32` tensor of shape (batch_size, max_len, max_len), representing
                the pointer probabilities.
        """
        encoder_outputs, hidden_state = self._encode(inputs, lengths=lengths)
        logits = self._decode(encoder_outputs, hidden_state, targets=targets, lengths=lengths)
        return logits

    def _encode(self, inputs, lengths=None):
        rnn_inputs = self.input_dropout(self.linear(inputs))
        encoder_outputs, hidden_state = utils.dynamic_rnn(self.encoder_rnn, rnn_inputs, lengths)
        return encoder_outputs, hidden_state

    def _inference_decode(self, encoder_outputs, hidden_state, lengths=None):
        batch_size, max_len, _ = encoder_outputs.size()
        if lengths is not None:
            padding_mask = utils.batch_sequence_mask(lengths)  # (batch_size, max_len)
        else:
            padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        input_index = torch.zeros(batch_size, 1, dtype=torch.int64)
        step_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        logits = []

        for i in range(max_len):
            rnn_input = utils.batch_gather(encoder_outputs, input_index)  # (batch_size, 1, hidden_size)
            decoder_rnn_output, hidden_state = self.decoder_rnn(rnn_input, hx=hidden_state)
            attention_mask = utils.combine_masks(
                step_mask[:, None, :],
                padding_mask[:, None, :],
                padding_mask[:, i: i + 1, None])
            logit = self.attention(decoder_rnn_output, encoder_outputs, mask=attention_mask)  # (batch_size, 1, max_len)
            logits.append(logit)
            input_index = logit.argmax(-1)
            step_mask = utils.batch_step_mask(step_mask, input_index.squeeze(1))
        return torch.cat(logits, dim=1)

    def _teacher_forcing(self, encoder_outputs, hidden_state, targets, lengths=None):
        # Teacher forcing mode.
        decoder_inputs = targets.roll(1, dims=1)
        decoder_rnn_inputs = utils.batch_gather(encoder_outputs, decoder_inputs)
        decoder_rnn_outputs, _ = utils.dynamic_rnn(
            self.decoder_rnn, decoder_rnn_inputs, lengths, hidden_state=hidden_state)

        if lengths is None:
            padding_mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            padding_mask = utils.batch_sequence_mask(lengths)  # (batch_size, max_len)
        steps_mask = utils.batch_steps_mask(targets)  # (batch_size, max_len, max_len)
        attention_mask = utils.combine_masks(
            padding_mask.unsqueeze(dim=-1),
            padding_mask.unsqueeze(dim=-2),
            steps_mask)
        logits = self.attention(decoder_rnn_outputs, encoder_outputs, mask=attention_mask)
        return logits

    def _decode(self, encoder_outputs, hidden_state, targets=None, lengths=None):
        if targets is None:
            return self._inference_decode(encoder_outputs, hidden_state, lengths=lengths)
        return self._teacher_forcing(encoder_outputs, hidden_state, targets, lengths=lengths)
