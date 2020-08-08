import abc
import numpy as np


from typing import List, Tuple


class Solution(object):
    def __init__(self, index, vector):
        """

        :param index: (..., num_nodes)
        :param vector: (..., num_nodes, len_vector)
        """
        self.index = np.array(index)
        self.vector = np.array(vector)

    @classmethod
    def from_structured_data(cls, data: Tuple[int, List[float]]):
        index, vector = zip(*data)
        return cls(index, vector)


class Region(object):
    def __init__(self, parameters):
        """

        :param parameters: (..., len_params)
        """
        self._parameters = np.array(parameters)

    @abc.abstractmethod
    def compute_waypoints(self, solution: Solution):
        raise NotImplementedError

    @property
    def shape(self):
        return self._parameters.shape[: -1]

    @property
    def dim(self):
        return self._parameters.ndim - 1


class EllipsoidRegion(Region):
    @classmethod
    def from_randomly_generated(cls, shape, low, high):
        params = []
        for l, h in zip(low, high):
            params.append(np.random.uniform(l, h, size=shape))
        return cls(np.stack(params, axis=-1))

    # parameters: [..., 6]
    def compute_waypoints(self, solution: Solution):
        index = solution.index
        vector = solution.vector



class TravelingSalesmanProblemWithNeighbor(object):
    def __init__(self, regions: Region):
        self._regions = regions

    @property
    def regions(self):
        return self._regions
