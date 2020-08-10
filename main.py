import problems.tspn as tspn
import rkga.rkga as rkga


def main():
    solver = rkga.RandomKeyGeneticAlgorithm()
    regions = tspn.EllipsoidRegion.from_randomly_generated((10,), [0.]*6, [1.]*6, 3)
    solver.compile(regions)
    solver.run()


if __name__ == '__main__':
    main()
