"""
Data format:
    inputs: (N, num_neighbors, param_dim)
    targets: (N, num_neighbors), (N, num_neighbors, vector_dim)

"""
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils import data as torch_data


class TSPDataset(torch_data.Dataset):
    def __init__(self, pattern, max_num_nodes=None):
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
        self.max_num_nodes = max_num_nodes

    def __getitem__(self, item):
        parameters = torch.tensor(np.array(pd.read_csv(self.param_files[item], header=None)), dtype=torch.float)
        rank = torch.tensor(np.array(pd.read_csv(self.rank_files[item], header=None))[0], dtype=torch.int)
        if self.max_num_nodes is None:
            return parameters, rank
        parameters = torch.nn.functional.pad(parameters, [0, 0, 0, self.max_num_nodes - parameters.shape[-2]])
        rank = torch.nn.functional.pad(rank, [0, self.max_num_nodes - rank.shape[-1]])
        return parameters, rank

    def __len__(self):
        return self.length


def main():
    pattern = '/Users/Tianziyang/Desktop/data/tsp/*'
    dataset = TSPDataset(pattern, max_num_nodes=50)
    print(dataset[0][0].shape, dataset[0][1].shape)


main()


