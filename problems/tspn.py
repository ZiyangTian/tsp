import abc

import numpy as np


class Node(object):
    def __init__(self, parameters):
        self._parameters = np.array(parameters)

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
    def shape(self):
        return self._parameters.shape[:-1]

    @property
    def param_dim(self):
        return self._parameters.shape[-1]

    def concat(self, other, axis=0):
        if axis >= len(self.shape):
            raise ValueError('Cannot concatenate nodes at the axis {}.'.format(axis))
        return type(self)(np.concatenate([self._parameters, other.parameters], axis=axis))


class Neighbor(Node):
    @property
    @abc.abstractmethod
    def vector_dim(self):
        raise NotImplementedError('Neighbor.vector_dim')

    @classmethod
    @abc.abstractmethod
    def compute_waypoints(cls, parameters, vector):
        raise NotImplementedError('Neighbor.compute_waypoints')


class EllipsoidNeighbor(Neighbor):
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
        # parameters: [..., len_params], vector: [..., len_vector]
        return parameters[..., :3] + parameters[..., 3:] * vector  # [..., 3]
