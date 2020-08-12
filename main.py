import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import problems.tspn as tspn
import rkga.solvers as solvers


def tsp_test():
    # data_file = os.path.join('data', 'problems', 'tsp_1.txt')
    # parameters = np.array(pd.read_csv(data_file, sep=' '))

    parameters = np.random.uniform(size=(30, 2))
    neighbors = tspn.EllipsoidNeighbor(parameters)
    solver = solvers.TSPSolver(population_size=10000)
    solver.compile(neighbors, traceback=True, use_cuda=True)
    print(solver.solve(1000, 100))
    exit()
    max_num_generations = 200
    for g in range(1, max_num_generations + 1):
        solver.evolute()
        objective = solver.optimal.objective
        if np.isnan(objective):
            break
        print(g, objective)
        optimal_rank = solver.optimal.rank
        optimal_rank = optimal_rank + [optimal_rank[0]]

        if g % 10 == 0:
            plt.clf()
            plt.close()
            plt.figure()
            plt.plot(
                neighbors.parameters[:, 0], neighbors.parameters[:, 1], 'ro', color='red')
            plt.plot(
                neighbors.parameters[:, 0][optimal_rank], neighbors.parameters[:, 1][optimal_rank], 'r-', color='blue')
            plt.pause(0.0001)
    print(solver.optimal)
    plt.ioff()
    plt.show()


def tspn_test():
    regions = tspn.EllipsoidNeighbor.from_randomly_generated(
        (20,),
        [0.]*6,
        [1., 1., 1., 0.1, 0.1, 0.1])
    solver = solvers.TSPNSolver(
        selection_proportion=0.45,
        population_size=10000)
    solver.compile(regions, use_cuda=False)
    solver.solve(1000, max_descending_generations=100)


if __name__ == '__main__':
    tsp_test()
