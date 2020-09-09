import torch

from models import attentions
from models import datasets
from models import tsp


def main():
    train_pattern = r'/Users/Tianziyang/Desktop/data/tsp/tsp.train'
    valid_pattern = r'/Users/Tianziyang/Desktop/data/tsp/tsp.valid'
    test_pattern = r'/Users/Tianziyang/Desktop/data/tsp/tsp.test'
    train_data_loader = datasets.TSPDataLoader(train_pattern, batch_size=128, shuffle=True, random_roll=True, random_flip=True)
    valid_data_loader = datasets.TSPDataLoader(valid_pattern, batch_size=128, shuffle=False)
    test_data_loader = datasets.TSPDataLoader(test_pattern, batch_size=128, shuffle=False)

    model = tsp.TSPModel(
        input_size=2,
        hidden_size=128,
        rnn_layers=1,
        rnn_type=torch.nn.GRU,
        input_dropout=0.1,
        encoder_rnn_dropout=0,
        decoder_rnn_dropout=0,
        attention_mechanism=attentions.LoungAttention,
        attention_hidden_size=None,
        eval_beam_size=4)

    model.compile(
        optimizer_obj=torch.optim.Adam,
        learning_rate=0.001)

    history = model.fit(train_data_loader, valid_data_loader, 100)
    print(history)

    # model.load_state_dict(torch.load('/Users/Tianziyang/Desktop/data/tsp/model.pkl'))
    # model.evaluate(datasets.TSPDataLoader(test_pattern, batch_size=128, shuffle=False))

    # debug_data = datasets.TSPDataLoader(test_pattern, batch_size=3, shuffle=False)
    # for data in debug_data:
    #     predictions, evaluations = model.test_step(data)
    #     print(data[1], data[2])
    #     print(predictions, evaluations)
    #     break


if __name__ == '__main__':
    main()
