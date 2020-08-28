import torch

from models import attentions
from models import datasets
from models import tsp


def main():
    train_pattern = r'E:\Programs\DataSets\tsp\tsp.train'
    valid_pattern = r'E:\Programs\DataSets\tsp\tsp.valid'
    test_pattern = r'E:\Programs\DataSets\tsp\tsp.test'
    train_data_loader = datasets.TSPDataLoader(train_pattern, batch_size=128, shuffle=True)
    valid_data_loader = datasets.TSPDataLoader(valid_pattern, batch_size=128, shuffle=False)
    test_data_loader = datasets.TSPDataLoader(test_pattern, batch_size=128, shuffle=False)

    model = tsp.TSPModel(
        param_dim=2,
        hidden_size=128,
        rnn_layers=1,
        encoder_dropout=0.1,
        encoder_rnn_type=torch.nn.GRU,
        encoder_rnn_dropout=0,
        decoder_dropout=0.1,
        decoder_rnn_type=torch.nn.GRU,
        decoder_rnn_dropout=0.,
        attention_mechanism=attentions.BahdanauAttention,
        optimizer_object=torch.optim.Adam,
        learning_rate=0.01,
        optimizer_kwargs={})

    history = model.fit(train_data_loader, valid_data_loader, 100)
    print(history)

    results = model.evaluate(test_data_loader)
    print(results)


if __name__ == '__main__':
    main()
