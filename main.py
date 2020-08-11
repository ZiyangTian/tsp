import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import problems.tspn as tspn
import rkga.rkga as rkga


def tsp_test():
    data_file = os.path.join('data', 'problems', 'tsp_1.txt')
    parameters = np.array(pd.read_csv(data_file, sep=' '))
    zeros = np.zeros([parameters.shape[0], 4], dtype=np.float)
    parameters = np.concatenate([parameters, zeros], axis=-1)

    neighbors = tspn.EllipsoidNeighbor(parameters)
    solver = rkga.RandomKeyGeneticAlgorithm(population_size=10000)
    solver.compile(neighbors, use_cuda=True)

    max_num_generations = 10000
    for g in range(1, max_num_generations + 1):
        solver.evolute()
        optimal_solution_value = solver.optimal['solution_value']
        if np.isnan(optimal_solution_value):
            break
        print(g, optimal_solution_value)
        optimal_rank = solver.optimal['rank']
        optimal_rank = optimal_rank[list(range(len(optimal_rank))) + [0]]

        if g % 10 == 0:
            plt.clf()
            plt.close()
            plt.figure()
            plt.plot(
                neighbors.parameters[:, 0], neighbors.parameters[:, 1], 'ro', color='red')
            plt.plot(
                neighbors.parameters[:, 0][optimal_rank], neighbors.parameters[:, 1][optimal_rank], 'r-', color='blue')
            plt.pause(0.0001)
    plt.ioff()
    plt.show()


def tspn_test():
    regions = tspn.EllipsoidNeighbor.from_randomly_generated(
        (20,),
        [0.]*6,
        [1., 1., 1., 0.1, 0.1, 0.1])
    solver = rkga.RandomKeyGeneticAlgorithm(
        selection_proportion=0.45,
        population_size=10000)
    solver.compile(regions, use_cuda=False)
    solver.run(1000, max_descending_generations=100)


if __name__ == '__main__':
    tsp_test()
