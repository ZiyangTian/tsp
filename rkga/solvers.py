import abc
import collections
import numpy as np
import torch

from problems import tspn


class _Config(object):
    selection_proportion = 0.35  # selection proportion
    crossover_proportion = 0.55  # crossover proportion
    crossover_threshold = 0.3  # crossover threshold
    mutation_prop = 0.01  # probability of gene mutation
    population_size = 10000  # population size.

    _epsilon = 1.e-6

    def __init__(self, **kwargs):
        self.selection_size = None
        self.crossover_size = None
        self.migration_size = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.selection_size = int(self.population_size * self.selection_proportion)  # Ms
        self.crossover_size = int(self.population_size * self.crossover_proportion)  # Mc
        self.migration_size = self.population_size - self.selection_size - self.crossover_size  # Mm


class _Solver(_Config):
    def __init__(self, **kwargs):
        super(_Solver, self).__init__(**kwargs)
        self._node_type = None
        self._node_data = None
        self._traceback = None
        self._param_dim = None
        self._num_nodes = None

        self._fractional_data = None  # MR
        self._fitness_data = None  # M
        self._optimal_index_data = None
        self._optimal_rank_data = None
        self._optimal_fitness_data = None
        self._optimal_objective_data = None
        self._num_generations = 0

        self._use_cuda = False

        self.__compiled = False

    def compile(self, nodes, traceback=True, use_cuda=False):
        """Compile the solver with a concrete problem.
        Arguments:
            nodes: A `Node` instance, representing the node parameters in the problem.
            traceback: A `bool`
            use_cuda: A `bool`, whether place tensors to GPU for acceleration.
        """
        self._use_cuda = use_cuda

        self._node_type = type(nodes)
        self._node_data = self._get_tensor(torch.tensor(nodes.parameters))  # RP
        self._traceback = traceback
        self._param_dim = nodes.parameters.shape[-1]
        self._num_nodes = nodes.shape[0]

        self.__compiled = True
        self.initialize()

    def solve(self, max_num_generations, max_descending_generations=None):
        self._compiled()

        global_optimal = None
        descending = 0
        last_fitness = -1
        for g in range(max_num_generations):
            self.evolute()
            optimal_fitness = self.optimal.fitness
            if optimal_fitness > last_fitness:
                descending = 0
                if global_optimal is None or optimal_fitness > global_optimal.fitness:
                    global_optimal = self.optimal
            else:
                descending += 1
                if max_descending_generations is not None and descending >= max_descending_generations:
                    break
            last_fitness = optimal_fitness
        return global_optimal

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
        self._fractional_data = self._initialize_individuals_randomly(self.population_size)
        self._fit()
        self._num_generations = 0

    def fitness_fn(self, objective):
        return 1. / (objective + self._epsilon)

    @property
    def solution(self):
        return collections.namedtuple('Solution', ('rank', 'objective', 'fitness'))

    @property
    def optimal(self):
        self._compiled()
        return self.solution(
            self._get_numpy(self._optimal_rank_data).tolist(),
            self._get_numpy(self._optimal_objective_data).item(),
            self._get_numpy(self._optimal_fitness_data).item())

    @abc.abstractmethod
    def _select(self):
        raise NotImplementedError('RandomKeyGeneticAlgorithmSolver._select')

    @abc.abstractmethod
    def _crossover(self):
        raise NotImplementedError('RandomKeyGeneticAlgorithmSolver._crossover')

    @abc.abstractmethod
    def _mutate(self):
        raise NotImplementedError('_Solver._mutate')

    @abc.abstractmethod
    def _migrate(self):
        raise NotImplementedError('_Solver._migrate')

    @abc.abstractmethod
    def _new_generation(self):
        raise NotImplementedError('_Solver._new_generation')

    @abc.abstractmethod
    def _fit(self):
        raise NotImplementedError('_Solver._fit')

    def _get_tensor(self, tensor):
        if not (self._use_cuda and torch.cuda.is_available()):
            return tensor
        return tensor.cuda()

    @staticmethod
    def _get_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        if tensor.is_cuda:
            return tensor.cpu().numpy()
        return tensor.numpy()

    @abc.abstractmethod
    def _initialize_individuals_randomly(self, n):
        raise NotImplementedError('_Solver._initialize_individuals_randomly')

    def _compiled(self):
        if not self.__compiled:
            raise ValueError('{} instance has not been complied.'.format(type(self).__name__))


class TSPSolver(_Solver):
    def __init__(self, **kwargs):
        super(TSPSolver, self).__init__(**kwargs)
        self._node_type = tspn.Node
        self._selected_fractional_temp_data = None
        self._crossover_fractional_temp_data = None
        self._migration_fractional_temp_data = None
        self.__compiled = False

    def compile(self, nodes, traceback=True, use_cuda=False):
        super(TSPSolver, self).compile(nodes, traceback=traceback, use_cuda=use_cuda)
        self.__compiled = True
        self.initialize()

    def initialize(self):
        super(TSPSolver, self).initialize()
        self._selected_fractional_temp_data = None
        self._crossover_fractional_temp_data = None
        self._migration_fractional_temp_data = None

    def _select(self):
        rank = torch.argsort(self._fitness_data, descending=True)
        self._selected_fractional_temp_data = self._fractional_data[rank][:self.selection_size]

    def _crossover(self):
        indices_1 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        indices_2 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        random_values = self._get_tensor(torch.rand(self.crossover_size, self._num_nodes))

        self._crossover_fractional_temp_data = torch.where(
            random_values > self.crossover_threshold,
            self._selected_fractional_temp_data[indices_1],
            self._selected_fractional_temp_data[indices_2])

    def _mutate(self):
        selected_mutation_random = self._get_tensor(torch.rand(self.selection_size, self._num_nodes))
        crossover_mutation_random = self._get_tensor(torch.rand(self.crossover_size, self._num_nodes))

        fractional = self._initialize_individuals_randomly(self.selection_size)
        self._selected_fractional_temp_data = torch.where(
            selected_mutation_random < self.mutation_prop,
            fractional, self._selected_fractional_temp_data)

        fractional = self._initialize_individuals_randomly(self.crossover_size)
        self._crossover_fractional_temp_data = torch.where(
            crossover_mutation_random < self.mutation_prop,
            fractional, self._crossover_fractional_temp_data)

    def _migrate(self):
        fractional = self._initialize_individuals_randomly(self.migration_size)
        self._migration_fractional_temp_data = fractional

    def _new_generation(self):
        self._fractional_data = torch.cat([
            self._selected_fractional_temp_data,
            self._crossover_fractional_temp_data,
            self._migration_fractional_temp_data],
            dim=0)

    def _fit(self):
        ranks = torch.argsort(self._fractional_data, dim=-1)
        waypoints = self._node_data[ranks]
        if self._traceback:
            tail = torch.cat([waypoints[..., 1:, :], waypoints[..., 0:1, :]], dim=-2)
            objective = (tail - waypoints).square().sum(dim=-1).sqrt().sum(dim=-1)
        else:
            objective = (waypoints[..., 1:, :] - waypoints[..., :-1, :]).square().sum(dim=-1).sqrt().sum(dim=-1)
        self._fitness_data = self.fitness_fn(objective)

        self._optimal_index_data = torch.argmax(self._fitness_data)
        self._optimal_rank_data = torch.argsort(self._fractional_data[self._optimal_index_data], dim=-1)
        self._optimal_fitness_data = self._fitness_data[self._optimal_index_data]
        self._optimal_objective_data = objective[self._optimal_index_data]

    def _initialize_individuals_randomly(self, n):
        fractional = torch.rand(n, self._num_nodes)  # nR
        fractional = self._get_tensor(fractional)
        return fractional


class TSPNSolver(TSPSolver):
    crossover_laplace_lambda = 0.
    crossover_laplace_mu = 1.

    def __init__(self, **kwargs):
        super(TSPNSolver, self).__init__(**kwargs)
        self.crossover_laplace = torch.distributions.Laplace(self.crossover_laplace_lambda, self.crossover_laplace_mu)

        self._node_type = None
        self._vector_dim = None
        self._vector_data = None

        self._selected_vector_temp_data = None
        self._crossover_vector_temp_data = None
        self._migration_vector_temp_data = None
        self.__compiled = False

    def compile(self, nodes, traceback=True, use_cuda=False):
        self._vector_dim = nodes.vector_dim
        super(TSPNSolver, self).compile(nodes, traceback=traceback, use_cuda=use_cuda)
        self._node_type = type(nodes)

    def initialize(self):
        self._compiled()
        self._fractional_data, self._vector_data = self._initialize_individuals_randomly(self.population_size)
        self._fit()
        self._num_generations = 0

        self._selected_fractional_temp_data = None
        self._selected_vector_temp_data = None
        self._crossover_fractional_temp_data = None
        self._crossover_vector_temp_data = None
        self._migration_fractional_temp_data = None
        self._migration_vector_temp_data = None

    @property
    def solution(self):
        return collections.namedtuple('Solution', ('rank', 'vector', 'objective', 'fitness'))

    @property
    def optimal(self):
        self._compiled()
        return self.solution(
            self._get_numpy(self._optimal_rank_data).tolist(),
            self._get_numpy(self._optimal_vector_data).tolist(),
            self._get_numpy(self._optimal_solution_value_data).item(),
            self._get_numpy(self._optimal_fitness_data).item())

    def _select(self):
        rank = torch.argsort(self._fitness_data, descending=True)
        self._selected_fractional_temp_data = self._fractional_data[rank][:self.selection_size]
        self._selected_vector_temp_data = self._vector_data[rank][:self.selection_size]  # MsRV

    def _crossover(self):
        indices_1 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        indices_2 = torch.randint(low=0, high=self.selection_size, size=(self.crossover_size,))
        random_values = self._get_tensor(torch.rand(self.crossover_size, self._num_nodes))

        self._crossover_fractional_temp_data = torch.where(
            random_values > self.crossover_threshold,
            self._selected_fractional_temp_data[indices_1],
            self._selected_fractional_temp_data[indices_2])

        laplace_values = self._get_tensor(self.crossover_laplace.sample((self.crossover_size,)))
        civ_1 = self._selected_vector_temp_data[indices_1]
        civ_2 = self._selected_vector_temp_data[indices_2]
        candidate_1 = civ_1 + laplace_values[:, None, None] * (civ_1 - civ_2)
        candidate_2 = civ_2 + laplace_values[:, None, None] * (civ_2 - civ_1)
        self._crossover_vector_temp_data = torch.where(
            random_values[:, :, None] > self.crossover_threshold, candidate_1, candidate_2)

    def _mutate(self):
        selected_mutation_random = self._get_tensor(torch.rand(self.selection_size, self._num_nodes))
        crossover_mutation_random = self._get_tensor(torch.rand(self.crossover_size, self._num_nodes))

        fractional, vector = self._initialize_individuals_randomly(self.selection_size)
        self._selected_fractional_temp_data = torch.where(
            selected_mutation_random < self.mutation_prop,
            fractional, self._selected_fractional_temp_data)
        self._selected_vector_temp_data = torch.where(
            selected_mutation_random[:, :, None] < self.mutation_prop,
            vector, self._selected_vector_temp_data)

        fractional, vector = self._initialize_individuals_randomly(self.crossover_size)
        self._crossover_fractional_temp_data = torch.where(
            crossover_mutation_random < self.mutation_prop,
            fractional, self._crossover_fractional_temp_data)
        self._crossover_vector_temp_data = torch.where(
            crossover_mutation_random[:, :, None] < self.mutation_prop,
            vector, self._crossover_vector_temp_data)

    def _migrate(self):
        fractional, vector = self._initialize_individuals_randomly(self.migration_size)
        self._migration_fractional_temp_data, self._migration_vector_temp_data = fractional, vector

    def _new_generation(self):
        super(TSPNSolver, self)._new_generation()
        self._vector_data = torch.cat([
                self._selected_vector_temp_data,
                self._crossover_vector_temp_data,
                self._migration_vector_temp_data],
            dim=0)

    def _fit(self):
        ranks = torch.argsort(self._fractional_data, dim=-1)
        sorted_nodes = self._node_data[ranks]
        waypoints = self._node_type.compute_waypoints(sorted_nodes, self._vector_data)
        if self._traceback:
            tail = torch.cat([waypoints[..., 1:, :], waypoints[..., 0:1, :]], dim=-2)
            objective = (tail - waypoints).square().sum(dim=-1).sqrt().sum(dim=-1)
        else:
            objective = (waypoints[..., 1:, :] - waypoints[..., :-1, :]).square().sum(dim=-1).sqrt().sum(dim=-1)

        self._fitness_data = self.fitness_fn(objective)
        self._optimal_index_data = torch.argmax(self._fitness_data)
        self._optimal_rank_data = torch.argsort(self._fractional_data[self._optimal_index_data])
        self._optimal_vector_data = self._vector_data[self._optimal_index_data]
        self._optimal_fitness_data = self._fitness_data[self._optimal_index_data]
        self._optimal_solution_value_data = 1. / self._optimal_fitness_data

    def _initialize_individuals_randomly(self, n):
        fractional = super(TSPNSolver, self)._initialize_individuals_randomly(n)
        vector = torch.randn(n, self._num_nodes, self._vector_dim).softmax(dim=-1)  # nRV
        vector = self._get_tensor(vector)
        return fractional, vector
