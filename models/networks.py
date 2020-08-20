import torch
from torch.utils import data as torch_data


from models import data


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, inputs, lengths):
        """

        :param inputs: max_len, batch_size, 2
        :param lengths: batch_size
        :return:
        """
        rnn_inputs = self.linear(inputs)
        rnn_inputs = self.dropout(rnn_inputs)
        packed_rnn_inputs = torch.nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths, enforce_sorted=False)
        packed_rnn_outputs, hidden = self.gru(packed_rnn_inputs)
        padded_rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_outputs)
        return padded_rnn_outputs, hidden


class BahdanauAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size)
        self.w2 = torch.nn.Linear(hidden_size, hidden_size)
        self.v = torch.nn.Linear(hidden_size, 1)

    def forward(self, query, values):
        # query hidden state shape == (1, batch_size, hidden size)
        # values shape == (max_len, batch_size, hidden size)
        query = query.permute(1, 0, 2)
        values = values.permute(1, 0, 2)

        # score shape == (batch_size, max_length, 1)
        score = self.v(torch.tanh(self.w1(query) + self.w2(values)))

        # attention_weights shape == (batch_size, max_len, 1)
        attention_weights = torch.nn.functional.softmax(score, dim=1)

        context_vector = attention_weights * values
        context_vector = context_vector.sum(dim=1)  # (batch_size, hidden_size)

        return attention_weights, context_vector


class Decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drouput):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(drouput)
        self.gru = torch.nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers)
        self.attention = BahdanauAttention(hidden_size)

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
    def __init__(self, input_size, hidden_size, encoder_rnn_layers, decoder_rnn_layers,
                 encoder_dropout=0., decoder_dropout=0.):
        super(PointerNetwork, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, encoder_rnn_layers, encoder_dropout)
        self.decoder = Decoder(input_size, hidden_size, decoder_rnn_layers, decoder_dropout)

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
