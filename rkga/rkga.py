import torch


class RKGAConfig(object):
    """
    population_size: M
    num_nodes: R
    param_dim: P
    vector_dim: V
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
        self._node_type = None
        self._node_data = None
        self._param_dim = None
        self._num_nodes = None
        self._vector_dim = None

        self._fractional_data = None  # MR
        self._vector_data = None  # MRV
        self._fitness_data = None  # M
        self._optimal_index_data = None
        self._optimal_rank_data = None
        self._optimal_vector_data = None
        self._optimal_fitness_data = None
        self._optimal_solution_value_data = None
        self._num_generations = 0

        self._use_cuda = False

        self.__compiled = False
        self.__selected_fractional_temp_data = None
        self.__selected_vector_temp_data = None
        self.__crossover_fractional_temp_data = None
        self.__crossover_vector_temp_data = None
        self.__migration_fractional_temp_data = None
        self.__migration_vector_temp_data = None

    def compile(self, nodes, use_cuda=False):
        """

        :param nodes:
        :param use_cuda:
        :return:
        """
        self._use_cuda = use_cuda

        self._node_type = type(nodes)
        self._node_data = self.maybe_cuda_tensor(torch.tensor(nodes.parameters))  # RP
        self._param_dim = nodes.parameters.shape[-1]
        self._num_nodes = nodes.shape[0]
        self._vector_dim = nodes.vector_dim

        self.__compiled = True
        self.initialize()

    def run(self, max_num_generations, max_descending_generations=None):
        self._compiled()
        descending = 0
        last_fitness = 0
        g = 0
        for g in range(max_num_generations):
            self.evolute()
            optimal_fitness = self.maybe_numpy_tensor(self._optimal_fitness_data)
            if optimal_fitness > last_fitness:
                descending = 0
            else:
                descending += 1
                if max_descending_generations is not None and descending >= max_descending_generations:
                    break
            last_fitness = optimal_fitness
        return g, self.maybe_numpy_tensor(self._optimal_solution_value_data)

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
    
    @property
    def optimal(self):
        self._compiled()
        return {
            'rank': self.maybe_numpy_tensor(self._optimal_rank_data),
            'vector': self.maybe_numpy_tensor(self._optimal_vector_data),
            'solution_value': self.maybe_numpy_tensor(self._optimal_solution_value_data),
            'fitness': self.maybe_numpy_tensor(self._optimal_fitness_data)}
    
    def _select(self):
        rank = torch.argsort(self._fitness_data, descending=True)
        self.__selected_fractional_temp_data = self._fractional_data[rank][:self.selection_size]
        self.__selected_vector_temp_data = self._vector_data[rank][:self.selection_size]  # MsRV

    def _crossover(self):
        indices_1 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        indices_2 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        random_values = self.maybe_cuda_tensor(torch.rand(self.crossover_size, self._num_nodes))

        self.__crossover_fractional_temp_data = torch.where(
            random_values > self.crossover_threshold,
            self.__selected_fractional_temp_data[indices_1],
            self.__selected_fractional_temp_data[indices_2])

        laplace_values = self.maybe_cuda_tensor(self.crossover_laplace.sample((self.crossover_size,)))
        civ_1 = self.__selected_vector_temp_data[indices_1]
        civ_2 = self.__selected_vector_temp_data[indices_2]
        candidate_1 = civ_1 + laplace_values[:, None, None] * (civ_1 - civ_2)
        candidate_2 = civ_2 + laplace_values[:, None, None] * (civ_2 - civ_1)
        self.__crossover_vector_temp_data = torch.where(
            random_values[:, :, None] > self.crossover_threshold, candidate_1, candidate_2)

    def _mutate(self):
        selected_mutation_random = self.maybe_cuda_tensor(torch.rand(self.selection_size, self._num_nodes))
        crossover_mutation_random = self.maybe_cuda_tensor(torch.rand(self.crossover_size, self._num_nodes))

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
        self._optimal_index_data = torch.argmax(self._fitness_data)
        self._optimal_rank_data = torch.argsort(self._fractional_data[self._optimal_index_data])
        self._optimal_vector_data = self._vector_data[self._optimal_index_data]
        self._optimal_fitness_data = self._fitness_data[self._optimal_index_data]
        self._optimal_solution_value_data = 1. / self._optimal_fitness_data

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
        fractional = torch.rand(n, self._num_nodes)  # nR
        vector = torch.randn(n, self._num_nodes, self._vector_dim).softmax(dim=-1)  # nRV
        fractional = self.maybe_cuda_tensor(fractional)
        vector = self.maybe_cuda_tensor(vector)
        return fractional, vector

    def _compute_fitness(self, fractional, vector):
        ranks = torch.argsort(fractional, dim=-1)
        sorted_nodes = self._node_data[ranks]
        waypoints = self._node_type.compute_waypoints(sorted_nodes, vector)

        tail = torch.cat([waypoints[..., 1:, :], waypoints[..., 0:1, :]], dim=-2)
        path_lengths = (tail - waypoints).square().sum(dim=-1).sqrt().sum(dim=-1)
        fitness = 1. / path_lengths
        return fitness

    def _compiled(self):
        if not self.__compiled:
            raise ValueError('Solver has not been complied.')
