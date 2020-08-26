"""Build TSP datasets and data loaders from data files."""
import glob
import numpy as np
import torch

from typing import Dict, Optional
from torch.utils import data as torch_data


class TSPDataset(torch_data.Dataset):
    """Build a TSP dataset from a data file. One data file contains multiple lines, each of which
    represents an example. The line contains the problems parameters and the target rank of one example,
    separated by a semicolon. The parameters with tensor shape (num_nodes, param_dim) have been
    flattened and are separated by commas. The ranks are also separated by commas. Since the solution
    route is a circle, the target rank can start at any node index. For consistency, all solution route
    are forced to start at the first node, that is, index zero. Thus, each line of a data file should be like:
        <float>,<float>,...,<float>;<index=0>,<index>,...,<index>
        Arguments:
            file (optional): Path to the data file. Defaults to `None`, that is, an empty dataset.
    """
    def __init__(self, file=None):
        # type: (TSPDataset, Optional[None, str]) -> None
        if file is None:
            self._lines = []
        else:
            with open(file, 'r') as f:
                self._lines = f.readlines()

    def __getitem__(self, item):
        # type: (TSPDataset, int) -> (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        Arguments:
            item: Index.
        Returns:
            parameters: Problem parameters of shape (num_nodes, param_dim).
            rank: Target rank of shape (num_nodes,)
            num_nodes: Number of nodes (scalar tensor), for batching and padding.
        """
        return self._parse_line(self._lines[item])

    def __len__(self):
        return len(self._lines)

    @staticmethod
    def _parse_line(line):
        # type: (str) -> (torch.Tensor, torch.Tensor, torch.Tensor)
        parameter_part, rank_part = line.strip().split(';')
        rank = np.array(list(map(int, rank_part.split(','))))
        num_nodes = len(rank)
        parameters = np.array(list(map(float, parameter_part.split(',')))).reshape((num_nodes, -1))
        return (torch.tensor(parameters, dtype=torch.float64),
                torch.tensor(rank, dtype=torch.int64),
                torch.tensor(num_nodes))


class TSPDataLoader(torch_data.DataLoader):
    """Build a TSP dataset from multiple data files.
        Arguments:
            pattern: Data file pattern.
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            kwargs: Parallel arguments for build the data loader, see `torch.utils.data.DataLoader`.
        Outputs:
            parameters: Problem parameters of shape (max_len, batch_size, param_dim).
            rank: Target ranks of shape (max_len, batch_size)
            num_nodes: Valid number of nodes of shape (batch_size,).
    """
    def __init__(self, pattern, batch_size=1, shuffle=False, **kwargs):
        # type: (TSPDataLoader, str, int, bool, Dict) -> None
        files = glob.glob(pattern)
        dataset = sum(map(TSPDataset, files), TSPDataset())

        def collate_fn(examples):
            parameters, ranks, lengths = zip(*examples)
            padded_parameters = torch.nn.utils.rnn.pad_sequence(parameters)
            padded_ranks = torch.nn.utils.rnn.pad_sequence(ranks)
            lengths = torch.stack(lengths)
            return padded_parameters, padded_ranks, lengths

        super(TSPDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)


def main():
    pattern = r'../demo/data.txt'
    dl = TSPDataLoader(pattern, batch_size=2)
    for x, y, z in dl:
        # print(x.shape, y.shape, z.shape)
        print(x, y, z)
        break


if __name__ == '__main__':
    main()
