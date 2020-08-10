import abc
import numpy as np


class Region(object):
    def __init__(self, parameters, len_vector):
        """

        :param parameters: (..., len_params)
        """
        self.parameters = np.array(parameters)
        self.len_vector = len_vector
        self.len_params = self.parameters.shape[-1]
        self.shape = self.parameters.shape[:-1]
        self.dim = len(self.shape)

    @classmethod
    def from_randomly_generated(cls, shape, low, high, len_vector):
        params = []
        for l, h in zip(low, high):
            params.append(np.random.uniform(l, h, size=shape))
        return cls(np.stack(params, axis=-1), len_vector)

    @classmethod
    @abc.abstractmethod
    def compute_waypoints_fn(cls, parameters, vector):
        raise NotImplementedError


class EllipsoidRegion(Region):
    @classmethod
    def compute_waypoints_fn(cls, parameters, vector):
        # parameters: [..., len_params], vector: [..., len_vector]
        return parameters[..., :3] + parameters[..., 3:] * vector  # [..., 3]
