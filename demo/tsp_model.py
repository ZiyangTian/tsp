import torch

from models import attentions
from models import datasets
from models import tsp


def main():
    train_pattern = r'E:\Programs\DataSets\tsp\train\tsp_all_len*.data'
    valid_pattern = r'E:\Programs\DataSets\tsp\train\tsp_10_test_exact.data'
    test_pattern = r'E:\Programs\DataSets\tsp\tsp.5-20.test'
    train_data_loader = datasets.TSPDataLoader(train_pattern, batch_size=128, shuffle=True, random_roll=True, random_flip=True)
    valid_data_loader = datasets.TSPDataLoader(valid_pattern, batch_size=128, shuffle=False)
    test_data_loader = datasets.TSPDataLoader(test_pattern, batch_size=128, shuffle=False)

    model = tsp.TSPModel(
        2,
        256,
        128,
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
        pointer_attention_dropout=0.,)

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
