"""Build TSP datasets and data loaders from data files."""
import glob
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
    Each example of this dataset is a tuple of parameters, rank, and number of nodes. The parameters are drawn
    from the left side content of the semicolon and the reshaped. The rank is drawn from the right side. Besides,
    the original rank begins with index zero, but we want to make a prediction to each next index. That is,
    for a rank <0, 3, 1, 4, 2>, we want to input a constant beginning index zero and output a prediction
    <3, 1, 4, 2, 0>. Thus we roll the original rank by 1 towards the right so as to match our model's target
    output.

        Arguments:
            file: A `str`, path to the data file. Defaults to `None`, that is, an empty dataset.
        Outputs:
            A `tuple` of (parameters, rank, num_nodes).
                parameters: A `torch.float32` tensor of shape (num_nodes, param_dim), the
                    problem parameters.
                rank: An `torch.int64` tensor of shape (num_nodes,), representing the ground
                    truth rank.
                num_nodes: A `torch.int64` scalar tensor, the number of nodes.
    """
    def __init__(self, file=None):
        # type: (TSPDataset, Optional[None, str]) -> None
        if file is None:
            lines = []
        else:
            with open(file, 'r') as f:
                lines = f.readlines()
        self._data = list(map(self._parse_line, lines))

    def __getitem__(self, item):
        # type: (TSPDataset, int) -> (torch.Tensor, torch.Tensor, torch.Tensor)
        return self._data[item]

    def __len__(self):
        return len(self._data)

    @staticmethod
    def _parse_line(line):
        # type: (str) -> (torch.Tensor, torch.Tensor, torch.Tensor)
        parameter_part, rank_part = line.strip().split(';')
        rank = torch.tensor(list(map(int, rank_part.split(','))), dtype=torch.int64).roll(-1)
        num_nodes = torch.tensor(rank.shape[0], dtype=torch.int64)
        parameters = torch.tensor(
            list(map(float, parameter_part.split(','))), dtype=torch.float).reshape(rank.shape[0], -1)
        return parameters, rank, num_nodes


class TSPDataLoader(torch_data.DataLoader):
    """Build a TSP dataset from multiple data files.
        Arguments:
            pattern: A `str`, data file pattern.
            batch_size: An `int`, batch size.
            shuffle: An `bool`, whether to shuffle the dataset.
            kwargs: Parallel arguments for build the data loader, see `torch.utils.data.DataLoader`.
        Outputs:
            A `tuple` of (parameters, ranks, num_nodes).
                parameters: A `torch.float32` tensor, the problem parameters, padded with zeros.
                    The shape is (max_len, batch_size, param_dim) if `time_major` is true, or else,
                    (batch_size, max_len, param_dim).
                rank: A `torch.int64` tensor, representing the ground truth rank, padded with
                    zeros. The shape is (max_len, batch_size) if `time_major` is true, or else,
                    (batch_size, max_len).
                num_nodes: A `torch.int64` tensor of shape (batch_size,), the number of nodes.
    """
    def __init__(self, pattern, batch_size=1, shuffle=False, **kwargs):
        # type: (TSPDataLoader, str, int, bool, Dict) -> None
        files = glob.glob(pattern)
        dataset = sum(map(TSPDataset, files), TSPDataset())

        def collate_fn(examples):
            parameters, ranks, num_nodes = zip(*examples)
            parameters = torch.nn.utils.rnn.pad_sequence(parameters, batch_first=True)
            ranks = torch.nn.utils.rnn.pad_sequence(ranks, batch_first=True)
            num_nodes = torch.stack(num_nodes)
            return parameters, ranks, num_nodes

        super(TSPDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
