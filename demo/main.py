import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import problems.tspn as tspn
import solvers.rkga as rkga
import solvers.opt as opt


def tsp_test():
    # data_file = os.path.join('demo', 'problems', 'tsp_1.txt')
    # parameters = np.array(pd.read_csv(data_file, sep=' '))

    # parameters = np.random.uniform(size=(30, 2))

    # parameters = np.array([
    #     0.607122, 0.664447, 0.953593, 0.021519, 0.757626, 0.921024, 0.586376, 0.433565, 0.786837, 0.052959,
    #     0.016088, 0.581436, 0.496714, 0.633571, 0.227777, 0.971433, 0.665490, 0.074331, 0.383556, 0.104392])
    # parameters = np.reshape(parameters, (10, 2))#  np.transpose(np.reshape(parameters, (2, 10)))  #   #
    # output = [1, 3, 8, 6, 10, 9, 5, 2, 4, 7, 1]
    # rank = [0, 2, 7, 5, 9, 8, 4, 1, 3, 6, 0]
    parameters = np.array([
        0.902421046498, 0.77621271719, 0.722727240262, 0.579024761326, 0.202222502748,
        0.629848925738, 0.577683327113, 0.943735341041, 0.387801872815, 0.846231151452,
        0.248295276407, 0.526557661494, 0.367962107849, 0.578749472622, 0.555896495344,
        0.251653475865, 0.907255613781, 0.245249027563, 0.52544531979, 0.115256640047,
        0.892999587695, 0.492378832333, 0.683712610357, 0.622637519591, 0.57168470001,
        0.764729790149, 0.455321988383, 0.383651641103, 0.114378142586, 0.0686257141516,
        0.880942807367, 0.565920630956, 0.0762609396025, 0.710417785386, 0.222822900992,
        0.043125608507, 0.226433818255, 0.568887954037, 0.858631853949, 0.949648342561,
        0.390812160258, 0.8303623789, 0.257238816645, 0.558283157573, 0.763412862654,
        0.158490634265, 0.624987630978, 0.390107045565, 0.130862344781, 0.510372583137,
        0.674928789702, 0.881428356182, 0.964366215304, 0.525331313649, 0.706494712117,
        0.449096582924, 0.788732677484, 0.526778478004, 0.318321798568, 0.3545657325,
        0.607914569902, 0.670640404295, 0.070789142522, 0.556820580678, 0.809475252633,
        0.394581642841, 0.317719793413, 0.870039653289, 0.375853198069, 0.19272404987,
        0.342342952722, 0.821914770523, 0.589592002558, 0.210835572805, 0.748900316792,
        0.84695495136, 0.516684883833, 0.801904939729, 0.508203739261, 0.432345157039,
        0.936113150967, 0.636238987707, 0.39863340474, 0.0702509602592, 0.208057203091,
        0.947559363155, 0.0211812129964, 0.352147795879, 0.674019523967, 0.0999767914088,
        0.380454813888, 0.440897245531, 0.0485243910601, 0.862506649068, 0.620546948127,
        0.545954375688, 0.514113046765, 0.66313845183, 0.350561488721, 0.000114044320017])
    parameters = np.reshape(parameters, (50, 2))
    rank = np.array([
        1, 41, 27, 11, 16, 29, 2, 12, 31, 49, 13, 39, 5, 21, 36, 34, 43, 47, 17, 44, 32, 25, 3, 19, 22,
        6, 7, 48, 28, 33, 9, 23, 45, 10, 37, 8, 24, 40, 14, 46, 30, 35, 42, 50, 18, 15, 4, 26, 38, 20, 1]) - 1

    neighbors = tspn.EllipsoidTSPN(parameters)
    solver = rkga.TSPSolver(population_size=10000)
    solver.compile(neighbors, traceback=True, use_cuda=True)
    # print(solver.solve(1000, 100))
    # exit()
    max_num_generations = 300
    for g in range(1, max_num_generations + 1):
        solver.evolute()
        objective = solver.optimal.objective
        if np.isnan(objective):
            break
        print(g, objective)
        optimal_rank = solver.optimal.rank
        optimal_rank = optimal_rank + [optimal_rank[0]]

    plt.clf()
    plt.close()
    plt.figure()
    plt.plot(
        neighbors.parameters[:, 0], neighbors.parameters[:, 1], 'ro', color='red')
    plt.plot(
        neighbors.parameters[:, 0][optimal_rank], neighbors.parameters[:, 1][optimal_rank], 'r-', color='blue')
    # plt.plot(
    #     neighbors.parameters[:, 0][rank], neighbors.parameters[:, 1][rank], 'r-', color='green')
    plt.pause(0.001)
    print(solver.optimal)

    waypoints = neighbors.parameters[rank]

    print('using...')
    optimal_rank_opt = opt.opt2_search(neighbors.parameters, optimal_rank)
    plt.clf()
    plt.close()
    plt.figure()
    plt.plot(
        neighbors.parameters[:, 0], neighbors.parameters[:, 1], 'ro', color='red')
    plt.plot(
        neighbors.parameters[:, 0][optimal_rank_opt], neighbors.parameters[:, 1][optimal_rank_opt], 'r-', color='blue')

    plt.ioff()
    plt.show()


def opt_test():
    parameters = np.array([
        0.902421046498, 0.77621271719, 0.722727240262, 0.579024761326, 0.202222502748,
        0.629848925738, 0.577683327113, 0.943735341041, 0.387801872815, 0.846231151452,
        0.248295276407, 0.526557661494, 0.367962107849, 0.578749472622, 0.555896495344,
        0.251653475865, 0.907255613781, 0.245249027563, 0.52544531979, 0.115256640047,
        0.892999587695, 0.492378832333, 0.683712610357, 0.622637519591, 0.57168470001,
        0.764729790149, 0.455321988383, 0.383651641103, 0.114378142586, 0.0686257141516,
        0.880942807367, 0.565920630956, 0.0762609396025, 0.710417785386, 0.222822900992,
        0.043125608507, 0.226433818255, 0.568887954037, 0.858631853949, 0.949648342561,
        0.390812160258, 0.8303623789, 0.257238816645, 0.558283157573, 0.763412862654,
        0.158490634265, 0.624987630978, 0.390107045565, 0.130862344781, 0.510372583137,
        0.674928789702, 0.881428356182, 0.964366215304, 0.525331313649, 0.706494712117,
        0.449096582924, 0.788732677484, 0.526778478004, 0.318321798568, 0.3545657325,
        0.607914569902, 0.670640404295, 0.070789142522, 0.556820580678, 0.809475252633,
        0.394581642841, 0.317719793413, 0.870039653289, 0.375853198069, 0.19272404987,
        0.342342952722, 0.821914770523, 0.589592002558, 0.210835572805, 0.748900316792,
        0.84695495136, 0.516684883833, 0.801904939729, 0.508203739261, 0.432345157039,
        0.936113150967, 0.636238987707, 0.39863340474, 0.0702509602592, 0.208057203091,
        0.947559363155, 0.0211812129964, 0.352147795879, 0.674019523967, 0.0999767914088,
        0.380454813888, 0.440897245531, 0.0485243910601, 0.862506649068, 0.620546948127,
        0.545954375688, 0.514113046765, 0.66313845183, 0.350561488721, 0.000114044320017])
    parameters = np.reshape(parameters, (50, 2))

    problem = tspn.TSP(parameters, traceback=True)
    solver = rkga.TSPSolver(population_size=10000)
    solver.compile(problem, use_cuda=False)
    solution = solver.solve(1000, 100)
    print(solution)

    optimal_rank = solver.optimal.rank
    optimal_rank = optimal_rank + [optimal_rank[0]]

    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(parameters[:, 0][optimal_rank + [optimal_rank[0]]],
             parameters[:, 1][optimal_rank + [optimal_rank[0]]],
             'r-', color='blue')

    optimal_rank_opt, optimal_journey_opt = opt.opt2_search(parameters, optimal_rank)
    print('optimal_journey_opt=%f' % optimal_journey_opt)
    plt.figure()
    plt.plot(parameters[:, 0], parameters[:, 1], 'ro', color='red')
    plt.plot(
        parameters[:, 0][optimal_rank_opt], parameters[:, 1][optimal_rank_opt], 'r-', color='green')

    plt.ioff()
    plt.show()


def tspn_test():
    neighbors = tspn.EllipsoidTSPN.from_randomly_generated(
        (20,),
        [0.]*6,
        [1., 1., 1., 0.1, 0.1, 0.1])
    nodes = tspn.TSP(neighbors.parameters[:, :3])
    tsp_solver = rkga.TSPSolver(population_size=10000)
    tsp_solver.compile(nodes, use_cuda=False)
    tspn_solver = rkga.TSPNSolver(population_size=300, opt2_prop=0.1)
    tspn_solver.compile(neighbors, use_cuda=False)
    print(tsp_solver.solve(1000).objective)
    print(tspn_solver.solve(1000).objective)
    for _ in range(5):
        tspn_solver.initialize()
        print(tspn_solver.solve(1000).objective)


def data_test():
    parameters_file = '/Users/Tianziyang/Desktop/data/tsp/3.parameters'
    rank_file = '/Users/Tianziyang/Desktop/data/tsp/3.rank'

    parameters = np.array(pd.read_csv(parameters_file, header=None))
    rank = list(np.array(pd.read_csv(rank_file, header=None))[0])
    rank = rank + [rank[0]]

    neighbors = tspn.EllipsoidTSPN(parameters)
    plt.figure()
    plt.plot(
        neighbors.parameters[:, 0], neighbors.parameters[:, 1], 'ro', color='red')
    plt.plot(
        neighbors.parameters[:, 0][rank], neighbors.parameters[:, 1][rank], 'r-', color='blue')
    print('...')
    # plt.plot(
    #     neighbors.parameters[:, 0][rank], neighbors.parameters[:, 1][rank], 'r-', color='green')
    plt.pause(0.1)
    waypoints = neighbors.parameters[rank]
    print(np.sqrt(np.square((waypoints[1:] - waypoints[:-1])).sum(-1)).sum())
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import data.generator
    data.generator.main()
