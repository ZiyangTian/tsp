"""Pointer networks."""
import torch

from torch.nn.utils import rnn as rnn_utils

from models import data
from models import attention


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
    def __init__(self, input_size, rnn, attention_mechanism, dropout=0.):
        # type: (PointerDecoder, int, torch.nn.RNNBase, attention.Attention, float) -> None
        super(PointerDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = rnn.hidden_size

        self.linear = torch.nn.Linear(input_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = attention_mechanism
        self.rnn = rnn

    def forward(self, hidden, encoder_outputs, inputs):
        """
        :param inputs: (batch_size, input_size) for teacher forcing,
            or (max_len, batch_size, input_size) for inference.
        :param hidden: num_layers, batch_size, hidden_size
        :param encoder_outputs: max_len, batch_size, hidden_size
        :return:
        """
        # hidden = hidden.mean(dim=0, keepdim=True)
        attention_weights, context_vector = self.attention(hidden.mean(dim=0, keepdim=True), encoder_outputs)
        output = attention_weights  # (batch_size, max_len, 1)

        if inputs.ndim == 3:
            output = torch.argmax(torch.squeeze(output, dim=-1), dim=1)
            inputs, output = torch.broadcast_tensors(inputs.permute(1, 0, 2), output[:, None, None])
            inputs = torch.gather(inputs, 1, output.to(dtype=torch.int64))[:, 0, :]
        else:
            inputs = inputs[None, ...]
        rnn_inputs = self.linear(inputs)
        rnn_inputs = self.dropout(rnn_inputs)
        _, hidden = self.gru(torch.cat([rnn_inputs, context_vector[None, ...]], dim=-1), hidden)
        return output, hidden


class PointerNetwork(torch.nn.Module):
    def __init__(self, encoder, decoder):
        # type: (PointerNetwork, PointerEncoder, PointerDecoder) -> None
        super(PointerNetwork, self).__init__()
        self.input_size = encoder.input_size
        self.hidden_size = encoder.hidden_size
        if self.hidden_size != decoder.hidden_size:
            raise ValueError

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, lengths, targets=None):
        """

        :param inputs: max_len, batch_size, 2
        :param lengths: batch_size
        :param targets: max_len, batch_size
        :return:
        """
        max_len, batch_size, input_size = inputs.shape
        encoder_outputs, hidden = self.encoder(inputs, lengths)

        if self.training:  # teacher forcing
            inputs_, targets_ = torch.broadcast_tensors(inputs.permute(1, 0, 2), targets.permute(1, 0)[..., None])
            decoder_inputs = torch.gather(inputs_, 1, targets_.to(dtype=torch.int64)).permute(1, 0, 2)
            # max_len, batch_size, 2
        else:
            decoder_inputs = inputs

        decoder_outputs = []
        for i in range(max_len):
            if self.training:
                output, hidden = self.decoder(hidden, encoder_outputs, inputs=decoder_inputs[i])
            else:
                output, hidden = self.decoder(hidden, encoder_outputs, inputs=decoder_inputs)
            decoder_outputs.append(output)
        outputs = torch.cat(decoder_outputs, dim=-1)
        return outputs.permute(0, 2, 1)  # batch_size, source_max_len, target_max_len


def loss_fn(inputs, targets, length=None):
    """

    :param inputs: batch_size, source_max_len, target_max_len
    :param targets: max_len, batch_size
    :param length: batch_size
    :return:
    """
    max_len = targets.shape[0]
    mask = torch.where(torch.arange(max_len)[None, :] < length[:, None], torch.tensor(1.), torch.tensor(0.))
    loss = torch.nn.functional.cross_entropy(
        inputs.permute(0, 2, 1), targets.permute(1, 0).to(dtype=torch.int64), reduction='none') * mask
    loss = torch.mean(loss)
    return loss


def main():
    pattern = r'/Users/Tianziyang/Desktop/data/tsp/*'
    dl = data.TSPDataLoader(pattern, batch_size=5)
    for parameters, rank, lengths in dl:
        break

    model = PointerNetwork(2, 8, 2, 2, 0.1, 0.1)
    y = model(parameters, lengths, rank)
    loss = loss_fn(y, rank, lengths)
    print(y.shape, loss)


if __name__ == '__main__':
    main()
