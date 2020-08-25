"""
Data format:
    inputs: (N, num_neighbors, param_dim)
    targets: (N, num_neighbors), (N, num_neighbors, vector_dim)

"""
import glob
import random
import numpy as np
import pandas as pd
import torch

from torch.utils import data as torch_data


class TSPDataset(torch_data.Dataset):
    def __init__(self, pattern, shift_rank_randomly=True):
        data_files = glob.glob(pattern)
        param_files = list(filter(lambda f: f.endswith('.parameters'), data_files))
        param_prefix = list(map(lambda f: f[:-11], param_files))
        rank_files = list(filter(lambda f: f.endswith('.rank'), data_files))
        rank_prefix = list(map(lambda f: f[:-5], rank_files))
        if set(param_prefix) != set(rank_prefix):
            raise ValueError('Unmatched parameter files and rank files.')
        sorted_prefix = sorted(param_prefix)
        self.length = len(sorted_prefix)
        self.param_files = list(map(lambda p: p + '.parameters', sorted_prefix))
        self.rank_files = list(map(lambda p: p + '.rank', sorted_prefix))

        self.shift_rank_randomly = shift_rank_randomly

    def __getitem__(self, item):
        parameters = torch.tensor(np.array(pd.read_csv(self.param_files[item], header=None)), dtype=torch.float)
        rank = np.array(pd.read_csv(self.rank_files[item], header=None))[0].tolist()
        length = len(rank)
        if self.shift_rank_randomly:
            random_int = random.randint(0, length)
            rank = rank[random_int:] + rank[: random_int]
        rank = torch.tensor(rank, dtype=torch.int64)
        return parameters, rank, torch.tensor(length)

    def __len__(self):
        return self.length


class TSPDataLoader(torch_data.DataLoader):
    def __init__(self, pattern, shift_rank_randomly=True, batch_size=1, shuffle=False, **kwargs):
        dataset = TSPDataset(pattern, shift_rank_randomly=shift_rank_randomly)

        def collate_fn(examples):
            parameters, ranks, lengths = zip(*examples)
            padded_parameters = torch.nn.utils.rnn.pad_sequence(parameters)
            padded_ranks = torch.nn.utils.rnn.pad_sequence(ranks)
            lengths = torch.stack(lengths)
            return padded_parameters, padded_ranks, lengths

        super(TSPDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)


def main():
    pattern = r'/Users/Tianziyang/Desktop/data/tsp/*'
    dl = TSPDataLoader(pattern, batch_size=2)
    for x in dl:
        print(x)


if __name__ == '__main__':
    main()
