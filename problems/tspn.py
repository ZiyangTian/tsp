import abc

import numpy as np


class TSP(object):
    def __init__(self, parameters):
        self._parameters = np.array(parameters)
        if self._parameters.ndim != 2:
            raise ValueError('`parameters` must be ranked 2.')

    @classmethod
    def load_from(cls, file, load_fn=None, **kwargs):
        if load_fn is None:
            load_fn = np.load
        parameters = load_fn(file, **kwargs)
        return cls(parameters)

    @property
    def parameters(self):
        return self._parameters

    @property
    def num_nodes(self):
        return self._parameters.shape[0]

    @property
    def param_dim(self):
        return self._parameters.shape[1]

    def concat(self, other):
        return type(self)(np.concatenate([self._parameters, other.parameters], axis=0))


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
