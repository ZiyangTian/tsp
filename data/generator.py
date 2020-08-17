import os
import time
import threading
import numpy as np
import pandas as pd

from problems import tspn
from rkga import solvers as rkga_solvers


class DataGenerator(object):
    def __init__(self, num_problems, num_nodes,
                      dimension=None, min_val=None, max_val=None,
                      num_parallel_solvers=None,
                      max_num_generations=100,
                      max_descending_generations=1000,
                      save_in=None, title_start_from=0,
                      **kwargs):
        min_val = min_val or 0.
        if type(min_val) in {float, int}:
            min_val = [float(min_val)] * dimension
        self._min_val = min_val
        max_val = max_val or 1.
        if type(max_val) in {float, int}:
            max_val = [float(max_val)] * dimension
        self._max_val = max_val


class SolverRunner(threading.Thread):
    def __init__(self, compiled_solver, max_num_generations, max_descending_generations=None):
        threading.Thread.__init__(self)
        self.solver = compiled_solver
        self.max_num_generations = max_num_generations
        self.max_descending_generations = max_descending_generations
        self.solution = None

    def run(self):
        self.solution = self.solver.solve(self.max_num_generations, self.max_descending_generations)


def generate_tsp_data(num_problems, num_nodes,
                      dimension=None, min_val=None, max_val=None,
                      num_parallel_solvers=None,
                      max_num_generations=100,
                      max_descending_generations=1000,
                      save_in=None, title_start_from=0,
                      **kwargs):
    min_val = min_val or 0.
    if type(min_val) in {float, int}:
        min_val = [float(min_val)] * dimension
    max_val = max_val or 1.
    if type(max_val) in {float, int}:
        max_val = [float(max_val)] * dimension

    use_multiple_solvers = num_parallel_solvers is not None
    if use_multiple_solvers:
        solvers = [rkga_solvers.TSPSolver(**kwargs) for _ in range(num_parallel_solvers)]
    else:
        solvers = [rkga_solvers.TSPSolver(**kwargs)]

    for i in range(num_problems):
        if type(num_nodes) is not int:
            n = np.random.randint(*num_nodes)
        else:
            n = num_nodes
        parameters = []
        for _min, _max in zip(min_val, max_val):
            parameters.append(np.random.uniform(_min, _max, size=(n,)))
        if len(parameters) != dimension:
            raise ValueError('`dimension` must match the sizes of `min_val` and `max_val`.')

        parameters = np.stack(parameters, axis=-1)
        nodes = tspn.TSP(parameters)

        if use_multiple_solvers:
            solver_threads = []
            for s in solvers:
                s.compile(nodes, traceback=True, use_cuda=True)
                thread = SolverRunner(s, max_num_generations, max_descending_generations)
                thread.start()
                solver_threads.append(thread)
            for s in solver_threads:
                s.join()
            solution = max(map(lambda st: st.solution, solver_threads), key=lambda sl: sl.fitness)
        else:
            solver = next(solvers.__iter__())
            solver.compile(nodes, traceback=True, use_cuda=True)
            solution = solver.solve(max_num_generations, max_descending_generations=max_descending_generations)

        with open(os.path.join(save_in, '{}.rank'.format(title_start_from + i)), 'w') as f:
            f.write(','.join(map(str, solution.rank)))
        pd.DataFrame(parameters).to_csv(
            os.path.join(save_in, '{}.parameters'.format(title_start_from + i)), header=False, index=False)


def main():
    generate_tsp_data(
        2, (11, 21),
        dimension=2, min_val=None, max_val=None,
        num_parallel_solvers=3,
        max_num_generations=1000,
        max_descending_generations=100,
        save_in=r'/Users/Tianziyang/Desktop/data/tsp')


if __name__ == '__main__':
    main()
