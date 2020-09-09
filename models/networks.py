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

        self.encoder_linear = torch.nn.Linear(self.input_size, self.hidden_size)
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

    def forward(self, inputs, targets=None, lengths=None, beam_size=None):
        """Forward propagation function.
        Arguments:
            inputs: A `torch.float32` tensor of shape (batch_size, max_len, input_size), representing
                the input parameters.
            targets: A `torch.int64` tensor of shape (batch_size, max_len), representing the ground
                truth of output ranks. If specified, use teacher forcing.
            lengths: A `torch.int64` tensor of shape (batch_size,), representing the valid sequence
                lengths of each example.
            beam_size:
        Returns:
            logits: A `torch.float32` tensor of shape (batch_size, max_len, max_len), representing
                the pointer probabilities.
        """
        encoder_inputs = self.input_dropout(self.encoder_linear(inputs))
        encoder_outputs, hidden_state = utils.dynamic_rnn(self.encoder_rnn, encoder_inputs, lengths)
        if targets is None:
            return self._inference(encoder_outputs, hidden_state, lengths=lengths, inputs=encoder_inputs, beam_size=beam_size)
        return self._teacher_forcing(encoder_outputs, hidden_state, targets=targets, lengths=lengths, inputs=encoder_inputs)

    def _inference(self, encoder_outputs, hidden_state, lengths=None, inputs=None, beam_size=None):
        beam_size = beam_size or 1
        batch_size, max_len, _ = encoder_outputs.size()
        if lengths is not None:
            padding_mask = utils.batch_sequence_mask(lengths, max_len=max_len)  # (batch_size, max_len)
        else:
            padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

        input_index = torch.zeros(batch_size, 1, 1, dtype=torch.int64)  # (batch_size, beam_size, 1)
        step_mask = torch.ones(batch_size, 1, max_len, dtype=torch.bool)  # (batch_size, beam_size, max_len)
        hidden_state = hidden_state[..., None, :]  # (1, batch_size, beam_size, hidden_size)
        log_probs = torch.zeros(batch_size, 1, dtype=torch.float32)  # (batch_size, beam_size)
        predictions = torch.empty(batch_size, beam_size, 0, dtype=torch.int64)  # (batch_size, beam_size, len)

        for i in range(max_len):  # For i == 0, dim(beam_size) is 1, else 0.
            rnn_input = utils.batch_gather(inputs, input_index.view(batch_size, -1))  # (batch_size, beam_size, hidden_size)
            decoder_rnn_output, new_hidden_state = self.decoder_rnn(
                rnn_input.view(-1, 1, self.hidden_size),
                hx=hidden_state.view(1, -1, self.hidden_size))
            attention_mask = utils.combine_masks(  # (batch_size*beam_size, 1, max_len)
                step_mask[..., None, :],
                padding_mask[:, None, None, :],
                padding_mask[:, None, i: i + 1, None]).view(-1, 1, max_len)
            repeated_encoder_outputs = torch.stack(
                [encoder_outputs] * (decoder_rnn_output.shape[0] // batch_size),
                dim=1).view(-1, max_len, self.hidden_size)  # (batch_size*beam_size, max_len, hidden_size)
            logit = self.attention(  # (batch_size*beam_size, 1, max_len)
                decoder_rnn_output,  # (batch_size*beam_size, 1, hidden_size)
                repeated_encoder_outputs,
                mask=attention_mask)
            log_probs = log_probs[:, :, None] + torch.log(logit.softmax(-1).reshape(batch_size, -1, max_len))  # (batch_size, beam_size, max_len)
            preds = torch.topk(log_probs.view(batch_size, -1), beam_size, dim=-1)  # (batch_size, beam_size*max_len)

            input_index = (preds.indices % max_len)[:, :, None]  # (batch_size, beam_size, 1)
            selected_beams = preds.indices // max_len  # (batch_size, beam_size)
            hidden_state = torch.gather(
                hidden_state, -2,
                torch.stack([selected_beams[None, :, :]] * self.hidden_size, dim=-1))
            if i == 0:
                step_mask = torch.cat([step_mask] * beam_size, dim=-2)
                predictions = input_index
            else:
                step_mask = torch.gather(step_mask, -2, utils.stack_n(selected_beams, max_len, -1))
                predictions = torch.gather(predictions, -2, utils.stack_n(selected_beams, predictions.shape[-1], -1))
                predictions = torch.cat([predictions, input_index], dim=-1)
            step_mask = utils.batch_step_mask(  # (batch_size, beam_size, max_len)
                step_mask.view(-1, max_len), input_index.flatten()).reshape(batch_size, beam_size, -1)

            log_probs = preds.values
        pred_beam = log_probs.argmax(-1)  # (batch_size,)
        predictions = torch.gather(predictions, 1, torch.stack([pred_beam] * max_len, dim=-1)[:, None, :]).squeeze(1)
        return predictions  # (batch_size, max_len)

    def _teacher_forcing(self, encoder_outputs, hidden_state, targets, lengths=None, inputs=None):
        # Teacher forcing mode.
        decoder_inputs = targets.roll(1, dims=1)
        decoder_rnn_inputs = utils.batch_gather(inputs, decoder_inputs)
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
            steps_mask
        )
        logits = self.attention(decoder_rnn_outputs, encoder_outputs, mask=attention_mask)
        return logits
