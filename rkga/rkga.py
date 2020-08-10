import torch

from problems import tspn


class RKGAConfig(object):
    """
    population_size: M
    num_regions: R
    len_params: P
    len_vector: V
    """
    selection_proportion = 0.35
    crossover_proportion = 0.55
    crossover_threshold = 0.3
    crossover_laplace_lambda = 0.
    crossover_laplace_mu = 1.
    mutation_prop = 0.01

    population_size = 10000

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.selection_size = int(self.population_size * self.selection_proportion)  # Ms
        self.crossover_size = int(self.population_size * self.crossover_proportion)  # Mc
        self.crossover_laplace = torch.distributions.Laplace(self.crossover_laplace_lambda, self.crossover_laplace_mu)
        self.migration_size = self.population_size - self.selection_size - self.crossover_size  # Mm


class RandomKeyGeneticAlgorithm(RKGAConfig):
    def __init__(self, **kwargs):
        super(RandomKeyGeneticAlgorithm, self).__init__(**kwargs)
        self._region_type = None
        self._region_data = None
        self._len_params = None
        self._num_regions = None
        self._len_vector = None

        self._fractional_data = None  # MR
        self._vector_data = None  # MRV
        self._fitness_data = None  # M
        self._best_index_data = None
        self._best_rank_data = None
        self._best_vector_data = None
        self._best_fitness_data = None
        self._num_generations = 0

        self._use_cuda = False

        self.__compiled = False
        self.__selected_fractional_temp_data = None
        self.__selected_vector_temp_data = None
        self.__crossover_fractional_temp_data = None
        self.__crossover_vector_temp_data = None
        self.__migration_fractional_temp_data = None
        self.__migration_vector_temp_data = None

    def compile(self, regions: tspn.Region, use_cuda=False):
        """

        :param regions:
        :param use_cuda:
        :return:
        """
        self._use_cuda = use_cuda

        self._region_type = type(regions)
        self._region_data = self.maybe_cuda_tensor(torch.tensor(regions.parameters))  # RP
        self._len_params = regions.parameters.shape[-1]
        self._num_regions = regions.shape[0]
        self._len_vector = regions.len_vector

        self.__compiled = True
        self.initialize()

    def run(self, max_num_generations, max_descending_generations=None):
        self._compiled()
        descending = 0
        last_fitness = 0
        for g in range(max_num_generations):
            self.evolute()
            best_fitness = self.maybe_numpy_tensor(self._best_fitness_data)
            print(1./best_fitness)
            if best_fitness > last_fitness:
                descending = 0
            else:
                descending += 1
                if max_descending_generations is not None and descending >= max_descending_generations:
                    return g, best_fitness
            last_fitness = best_fitness
        return g, best_fitness

    def evolute(self):
        self._compiled()
        self._select()
        self._crossover()
        self._mutate()
        self._migrate()
        self._new_generation()
        self._fit()

    def initialize(self):
        self._compiled()
        self._fractional_data, self._vector_data = self._initialize_individuals_randomly(self.population_size)
        self._fit()
        self._num_generations = 0

        self.__selected_fractional_temp_data = None
        self.__selected_vector_temp_data = None
        self.__crossover_fractional_temp_data = None
        self.__crossover_vector_temp_data = None
        self.__migration_fractional_temp_data = None
        self.__migration_vector_temp_data = None

    def _select(self):
        rank = torch.argsort(self._fitness_data, descending=True)
        self.__selected_fractional_temp_data = self._fractional_data[rank][:self.selection_size]
        self.__selected_vector_temp_data = self._vector_data[rank][:self.selection_size]  # MsRV

    def _crossover(self):
        indices_1 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        indices_2 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        random_values = self.maybe_cuda_tensor(torch.rand(self.crossover_size, self._num_regions))
        self.__crossover_fractional_temp_data = torch.where(
            random_values > self.crossover_threshold,
            self.__selected_fractional_temp_data[indices_1],
            self.__selected_fractional_temp_data[indices_2])

        # ...?
        laplace_values = self.maybe_cuda_tensor(self.crossover_laplace.sample((self.crossover_size,)))
        civ_1 = self.__selected_vector_temp_data[indices_1]
        civ_2 = self.__selected_vector_temp_data[indices_2]
        self.__crossover_vector_temp_data = civ_1 + laplace_values[:, None, None] * (civ_1 - civ_2)

    def _mutate(self):
        selected_mutation_random = self.maybe_cuda_tensor(torch.rand(self.selection_size, self._num_regions))
        crossover_mutation_random = self.maybe_cuda_tensor(torch.rand(self.crossover_size, self._num_regions))

        fractional, vector = self._initialize_individuals_randomly(self.selection_size)
        self.__selected_fractional_temp_data = torch.where(
            selected_mutation_random < self.mutation_prop,
            fractional, self.__selected_fractional_temp_data)
        self.__selected_vector_temp_data = torch.where(
            selected_mutation_random[:, :, None] < self.mutation_prop,
            vector, self.__selected_vector_temp_data)

        fractional, vector = self._initialize_individuals_randomly(self.crossover_size)
        self.__crossover_fractional_temp_data = torch.where(
            crossover_mutation_random < self.mutation_prop,
            fractional, self.__crossover_fractional_temp_data)
        self.__crossover_vector_temp_data = torch.where(
            crossover_mutation_random[:, :, None] < self.mutation_prop,
            vector, self.__crossover_vector_temp_data)

    def _migrate(self):
        fractional, vector = self._initialize_individuals_randomly(self.migration_size)
        self.__migration_fractional_temp_data, self.__migration_vector_temp_data = fractional, vector

    def _new_generation(self):
        self._fractional_data = torch.cat([
                self.__selected_fractional_temp_data,
                self.__crossover_fractional_temp_data,
                self.__migration_fractional_temp_data],
            dim=0)
        self._vector_data = torch.cat([
                self.__selected_vector_temp_data,
                self.__crossover_vector_temp_data,
                self.__migration_vector_temp_data],
            dim=0)

    def _fit(self):
        self._fitness_data = self._compute_fitness(self._fractional_data, self._vector_data)
        self._best_index_data = torch.argmax(self._fitness_data)
        self._best_rank_data = torch.argsort(self._fractional_data[self._best_index_data])
        self._best_vector_data = self._vector_data[self._best_index_data]
        self._best_fitness_data = self._fitness_data[self._best_index_data]

    def maybe_cuda_tensor(self, tensor):
        if not (self._use_cuda and torch.cuda.is_available()):
            return tensor
        return tensor.cuda()

    @staticmethod
    def maybe_numpy_tensor(tensor):
        if tensor.is_cuda:
            return tensor.cpu().numpy()
        return tensor.numpy()

    def _initialize_individuals_randomly(self, n):
        fractional = torch.rand(n, self._num_regions)  # nR
        vector = torch.randn(n, self._num_regions, self._len_vector).softmax(dim=-1)  # nRV
        fractional = self.maybe_cuda_tensor(fractional)
        vector = self.maybe_cuda_tensor(vector)
        return fractional, vector

    def _compute_fitness(self, fractional, vector):
        ranks = torch.argsort(fractional, dim=-1)
        sorted_regions = self._region_data[ranks]
        waypoints = self._region_type.compute_waypoints_fn(sorted_regions, vector)

        tail = torch.cat([waypoints[..., 1:, :], waypoints[..., 0:1, :]], dim=-2)
        path_lengths = (tail - waypoints).square().sum(dim=-1).sqrt().sum(dim=-1)
        fitness = 1. / path_lengths
        return fitness

    def _compiled(self):
        if not self.__compiled:
            raise ValueError('Solver has not been complied.')
