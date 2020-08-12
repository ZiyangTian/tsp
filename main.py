import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import problems.tspn as tspn
import rkga.solvers as solvers


def tsp_test():
    data_file = os.path.join('demo', 'problems', 'tsp_1.txt')
    parameters = np.array(pd.read_csv(data_file, sep=' '))

    # parameters = np.random.uniform(size=(30, 2))
    neighbors = tspn.EllipsoidNeighbor(parameters)
    solver = solvers.TSPSolver(population_size=10000)
    solver.compile(neighbors, traceback=True, use_cuda=True)
    # print(solver.solve(1000, 100))
    # exit()
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
            plt.pause(0.5)
    print(solver.optimal)
    plt.ioff()
    plt.show()


def tspn_test():
    neighbors = tspn.EllipsoidNeighbor.from_randomly_generated(
        (20,),
        [0.]*6,
        [1., 1., 1., 0.1, 0.1, 0.1])
    nodes = tspn.Node(neighbors.parameters[:, :3])
    tsp_solver = solvers.TSPSolver(population_size=1000)
    tsp_solver.compile(nodes, use_cuda=False)
    tspn_solver = solvers.TSPNSolver(population_size=1000)
    tspn_solver.compile(neighbors, use_cuda=False)
    print(tsp_solver.solve(1000).objective)
    print(tspn_solver.solve(1000).objective)
    for _ in range(5):
        tspn_solver.initialize()
        print(tspn_solver.solve(1000).objective)


if __name__ == '__main__':
    tspn_test()
