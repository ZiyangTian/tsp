import torch
from torch.utils import data as torch_data


from models import data


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, inputs, hidden, mask):
        outputs = self.linear(inputs.permute(1, 0, 2))
        outputs, hidden = self.gru(outputs)
        parameters = parameters


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def main():
    pattern = r'E:\Programs\DataSets\tsp\*'
    dataset = data.TSPDataset(pattern, max_num_nodes=50, shift_rank_randomly=True)
    data_loader = torch_data.DataLoader(dataset, batch_size=4, shuffle=True)
    for parameters, rank, mask in data_loader:
        break
