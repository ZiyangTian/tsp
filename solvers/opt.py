import os
import random
import numpy as np
import pandas as pd


def opt2_search(parameters, rank, traceback=True):
    """

    :param parameters: np.array
    :param rank: List
    :param traceback:
    :return:
    """
    difference_matrix = parameters[None, :, :] - parameters[:, None, :]
    difference_matrix = np.sqrt(np.sum(np.square(difference_matrix), axis=-1))

    optimal_journey = float('inf')
    optimal_rank = rank
    for i in range(len(rank)):
        for j in range(i):
            exchanged_rank = rank[:j] + list(reversed(rank[j:i])) + rank[i:]

            if traceback:
                node_index = exchanged_rank + [exchanged_rank[0]]
            else:
                node_index = exchanged_rank
            distance_index = list(zip(node_index[1:], node_index[:-1]))
            distance = np.array(list(map(lambda x: difference_matrix[x], distance_index)))
            journey = np.sum(distance)
            if journey < optimal_journey:
                optimal_rank = exchanged_rank
                optimal_journey = journey
    return optimal_rank, optimal_journey


def main():
    data_file = os.path.join('demo', 'problems', 'tsp_2.txt')
    parameters = np.array(pd.read_csv(data_file, sep=' ', header=None))
