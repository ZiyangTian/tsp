import problems.tspn as tspn
import rkga.rkga as rkga


def main():
    regions = tspn.EllipsoidRegion.from_randomly_generated(
        (20,), [0.]*6, [1., 1., 1., 0.1, 0.1, 0.1]*6, 3)
    solver = rkga.RandomKeyGeneticAlgorithm()
    solver.compile(regions, use_cuda=False)
    solver.run(10000)


if __name__ == '__main__':
    main()
