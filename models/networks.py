"""Pointer networks."""
import torch

from models import attentions
from models import utils


class PointerSequenceToSequence(torch.nn.Module):
    def __init__(self,
                 input_size,
                 dense_size,
                 hidden_size,
                 rnn_type=torch.nn.GRU,
                 rnn_layers=1,
                 encoder_input_dropout=0.,
                 decoder_input_dropout=0.,
                 encoder_rnn_dropout=0.,
                 decoder_rnn_dropout=0.,
                 output_dropout=0.,
                 seq2seq_attention_mechanism=attentions.BahdanauAttention,
                 seq2seq_attention_hidden_size=None,
                 seq2seq_attention_dropout=0.,
                 pointer_attention_mechanism=attentions.BahdanauAttention,
                 pointer_attention_hidden_size=None,
                 pointer_attention_dropout=0.):
        super(PointerSequenceToSequence, self).__init__()
        self.input_size = input_size
        self.dense_size = dense_size
        self.hidden_size = hidden_size

        self.input_dense = torch.nn.Linear(self.input_size, self.dense_size)
        self.encoder_input_dropout = torch.nn.Dropout(encoder_input_dropout)
        self.decoder_input_dropout = torch.nn.Dropout(decoder_input_dropout)
        self.encoder_rnn = rnn_type(
            self.dense_size,
            self.hidden_size,
            num_layers=rnn_layers,
            dropout=encoder_rnn_dropout,
            batch_first=True)
        self.decoder_rnn = rnn_type(
            self.dense_size + self.hidden_size,
            self.hidden_size,
            num_layers=rnn_layers,
            dropout=decoder_rnn_dropout,
            batch_first=True)
        self.seq2seq_attention = seq2seq_attention_mechanism(
            seq2seq_attention_hidden_size or self.hidden_size,
            query_hidden_size=rnn_layers * self.hidden_size,
            dropout=seq2seq_attention_dropout)
        self.output_dense = torch.nn.Linear(self.hidden_size, self.dense_size)
        self.output_dropout = torch.nn.Dropout(output_dropout)
        self.pointer_attention = pointer_attention_mechanism(
            pointer_attention_hidden_size or self.dense_size,
            dropout=pointer_attention_dropout)

    def forward(self, inputs, lengths=None, targets=None, teacher_forcing_prob=0.5):
        """Forward propagation function.
        Arguments:
            inputs: A `torch.float32` tensor of shape (batch_size, max_len, input_size), representing
                the input parameters.
            lengths: A `torch.int64` tensor of shape (batch_size,), representing the valid sequence
                lengths of each example.
            targets: A `torch.int64` tensor of shape (batch_size, max_len), representing the ground
                truth of output ranks. If specified, use teacher forcing.
            teacher_forcing_prob:
        Returns:
            logits: A `torch.float32` tensor of shape (batch_size, max_len, max_len), representing
                the pointer probabilities.
        """
        batch_size, max_len, _ = inputs.shape

        dense_inputs = self.input_dense(inputs)  # N, L, D
        encoder_inputs = self.encoder_input_dropout(dense_inputs)  # N, L, D
        encoder_outputs, hidden_state = utils.dynamic_rnn(self.encoder_rnn, encoder_inputs, lengths=lengths)
        # N, L, H; num_layers, N, H

        if lengths is None:
            padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        else:
            padding_mask = utils.batch_sequence_mask(lengths, max_len=max_len)
        logits, predictions = self._decode(dense_inputs, encoder_outputs, hidden_state, padding_mask,
                                           targets=targets, teacher_forcing_prob=teacher_forcing_prob)
        return logits, predictions

    def _decode(self, dense_inputs, encoder_outputs, hidden_state, padding_mask,
                targets=None, teacher_forcing_prob=0.5):
        batch_size, max_len, _ = dense_inputs.shape
        decoder_input_index = torch.zeros(batch_size, 1, dtype=torch.int64)  # N, 1
        step_mask = torch.ones(batch_size, max_len, dtype=torch.int64)  # N, L
        logits, predictions = [], []

        for i in range(max_len):
            decoder_input = utils.batch_gather(dense_inputs, decoder_input_index)  # N, 1, D
            context, _ = self.seq2seq_attention(  # N, 1, H
                hidden_state.permute(1, 0, 2).reshape(batch_size, 1, -1),
                encoder_outputs,
                encoder_outputs,
                mask=utils.combine_masks(padding_mask[:, i: i + 1, None], padding_mask.unsqueeze(1)))
            concatenated_input = torch.cat([context, decoder_input], dim=-1)  # N, 1, H+D
            decoder_output, hidden_state = self.decoder_rnn(concatenated_input, hidden_state)  # N, 1, H; ...
            pointer_mask = utils.combine_masks(  # N, 1, L
                step_mask.unsqueeze(1), padding_mask[:, i: i + 1, None], padding_mask.unsqueeze(1))
            logit, prediction = self._pointer(dense_inputs, decoder_output, mask=pointer_mask)  # N, 1, L; N, 1

            logits.append(logit)
            predictions.append(prediction)
            if teacher_forcing_prob == 1. or torch.rand(()) < teacher_forcing_prob:
                decoder_input_index = targets[i]
            else:
                decoder_input_index = prediction
            step_mask = utils.batch_step_mask(step_mask, decoder_input_index.squeeze(-1))

        logits = torch.cat(logits, dim=1)
        predictions = torch.cat(predictions, dim=1)
        return logits, predictions

    def _pointer(self, dense_inputs, decoder_outputs, mask=None):
        dense_outputs = self.output_dense(self.output_dropout(decoder_outputs))  # N, L, D
        logits = self.pointer_attention(dense_outputs, dense_inputs, mask=mask)
        predictions = logits.argmax(-1)
        return logits, predictions


if __name__ == '__main__':
    model = PointerSequenceToSequence(2, 8, 6, rnn_layers=2, pointer_attention_mechanism=attentions.LoungAttention)
    inputs = torch.randn(3, 5, 2)
    lengths = torch.tensor([4, 5, 3])
    targets = torch.tensor([
        [3, 1, 2, 0, 0],
        [3, 4, 1, 2, 0],
        [2, 1, 0, 0, 0]
    ])
    logits, predictions = model(inputs, lengths=lengths, targets=targets)
    print(logits.shape, predictions.shape)

