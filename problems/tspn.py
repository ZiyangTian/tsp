import abc

import numpy as np
import torch


class TSPNodes(object):
    def __init__(self, parameters):
        self._parameters = np.array(parameters)

    @property
    def parameters(self):
        return self._parameters

    @property
    def shape(self):
        return np.shape(self._parameters.shape)[:-1]

    @property
    def param_dim(self):
        return self._parameters.shape[-1]

    def concat(self, other):
        return type(self)(np.concatenate([self._parameters, other.parameters], axis=0))

    @classmethod
    def load_from(cls, file, load_fn=None, **kwargs):
        if load_fn is None:
            load_fn = np.load
        parameters = load_fn(file, **kwargs)
        return cls(parameters)


class TSP(TSPNodes):
    def __init__(self, parameters, traceback=True):
        super(TSP, self).__init__(parameters)
        self.traceback = traceback
        if self._parameters.ndim != 2:
            raise ValueError('`parameters` must be ranked 2.')

    @property
    def num_nodes(self):
        return self._parameters.shape[0]


class GTSP(TSPNodes):
    def __init__(self, parameters, traceback=True):
        super(GTSP, self).__init__(parameters)
        self.traceback = traceback
        if self._parameters.ndim != 3:
            raise ValueError('`parameters` must be ranked 3.')
        self._nodes_mask = np.isnan(self._parameters[..., 0])  # True for mask.

    @property
    def num_regions(self):
        return self.shape[0]

    @property
    def max_nodes(self):
        return self.shape[1]

    def num_nodes_in_region(self, n):
        return self.max_nodes - np.sum((self._nodes_mask[n]).astype(np.int))

    @classmethod
    def compute_waypoints(cls, parameters, index):
        # parameters: [num_regions, max_nodes, param_dim], index: [num_regions]
        return torch.gather(parameters, 1, index)


class TSPN(TSP):
    @property
    @abc.abstractmethod
    def vector_dim(self):
        raise NotImplementedError('TSPN.vector_dim')

    @classmethod
    @abc.abstractmethod
    def compute_waypoints(cls, parameters, vector):
        raise NotImplementedError('TSPN.compute_waypoints')


class EllipsoidTSPN(TSPN):
    @property
    def vector_dim(self):
        return 3

    @classmethod
    def from_randomly_generated(cls, shape, low, high):
        params = []
        for _l, _h in zip(low, high):
            params.append(np.random.uniform(_l, _h, size=shape))
        return cls(np.stack(params, axis=-1))

    @classmethod
    def compute_waypoints(cls, parameters, vector):
        # parameters: [num_nodes, len_params], vector: [num_nodes, len_vector]
        return parameters[..., :3] + parameters[..., 3:] * vector  # [num_nodes, 3]
