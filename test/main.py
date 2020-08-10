import problems.tspn as tspn
import rkga.rkga as rkga


def main():
    regions = tspn.EllipsoidRegion.from_randomly_generated(
        (20,), [0.]*6, [1., 1., 1., 0.1, 0.1, 0.1], 3)
    solver = rkga.RandomKeyGeneticAlgorithm(population_size=30)
    solver.compile(regions, use_cuda=False)
    solver.run(1000)


if __name__ == '__main__':
    main()
