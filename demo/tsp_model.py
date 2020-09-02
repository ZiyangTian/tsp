import torch

from models import attentions
from models import datasets
from models import tsp


def main():
    train_pattern = r'E:\Programs\DataSets\tsp\tsp.5-50.train'
    valid_pattern = r'E:\Programs\DataSets\tsp\tsp.5-50.valid'
    test_pattern = r'E:\Programs\DataSets\tsp\tsp.5-50.test'
    train_data_loader = datasets.TSPDataLoader(train_pattern, batch_size=128, shuffle=True)
    valid_data_loader = datasets.TSPDataLoader(valid_pattern, batch_size=32, shuffle=False)
    test_data_loader = datasets.TSPDataLoader(test_pattern, batch_size=32, shuffle=False)

    model = tsp.TSPModel(
        param_dim=2,
        hidden_size=64,
        rnn_layers=1,
        rnn_type=torch.nn.GRU,
        input_dropout=0.,
        encoder_rnn_dropout=0.,
        decoder_rnn_dropout=0.,
        attention_mechanism=attentions.LoungAttention,
        attention_hidden_size=None,
        optimizer_obj=torch.optim.Adam,
        learning_rate=0.001,
        optimizer_kwargs={})

    history = model.fit(train_data_loader, valid_data_loader, 100)
    print(history)

    results = model.evaluate(test_data_loader)
    print(results)


if __name__ == '__main__':
    main()
