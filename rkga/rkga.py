import torch

from python import tspn


class RKGAConfig(object):
    selection_proportion = 0.35
    crossover_proportion = 0.55
    crossover_threshold = 0.3
    mutation_prop = 0.01
    population_size = 100  # P

    individual_size = 100  # I
    dim = 3  # G
    region_param_size = 3  # R
    max_num_generations = 100

    # dimention: D

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.selection_size = int(self.population_size * self.selection_proportion)
        self.crossover_size = int(self.population_size * self.crossover_proportion)
        self.mutation_size = self.population_size - self.selection_size - self.crossover_size


class RandomKeyGeneticAlgorithm(RKGAConfig):
    def __init__(self, **kwargs):
        super(RandomKeyGeneticAlgorithm, self).__init__(**kwargs)
        self.problem = None

        self._problem_data = None  # torch.tensor(I, 6)
        self._individual_fractional_data = None  # torch.tensor(P, I)
        self._individual_vector_data = None  # torch.tensor(P, I, R)
        self._solution_data = None

    def compile(self, problem: tspn.TravelingSalesmanProblemWithNeighbor):
        self._individual_fractional_data = torch.rand(self.population_size, self.individual_size)
        self._individual_vector_data = torch.randn(  # ...
            self.population_size, self.individual_size, self.region_param_size).softmax(dim=-1)

    def evolute(self):
        pass

    def initialize(self):
        self._individual_fractional_data = torch.rand(self.population_size, self.individual_size)
        self._individual_vector_data = torch.randn(  # ...
            self.population_size, self.individual_size, self.dim).softmax(dim=-1)
        self._solution_data = self._individual_fractional_data.argsort(dim=-1)

    def compute_fitness(self):
        self.


