import numpy
import numpy as np

from helpers.helper import generate_risk_and_return_weights
from helpers.visualizer import plot_pareto_front

from warren_buffett import WarrenBuffett


class Instance:
    def __init__(self, cov_matrix, predictions):
        self.cov_matrix = cov_matrix
        self.predictions = predictions
        self.assets_number = len(predictions)


class ScalarizingFunction:
    def __init__(self, risk_factor, profit_factor):
        self.risk_factor = risk_factor
        self.profit_factor = profit_factor

    def __call__(self, individual: numpy.array, instance: Instance):
        return self.profit_factor * instance.predictions.dot(individual) - self.risk_factor * individual.dot(
            instance.cov_matrix).dot(individual)

    def __eq__(self, other):
        if isinstance(other, ScalarizingFunction):
            return (self.risk_factor, self.profit_factor) == (other.risk_factor, other.profit_factor)
        return False

    def __hash__(self):
        return hash((self.risk_factor, self.profit_factor))


# class Individual:
#     def __init__(self, weights):
#         self.weights = weights


class MOEADAlgorithm:
    def __init__(self, instance: Instance, **kwargs):
        self.instance = instance

        self.population_size = kwargs.get('population_size', 10)
        self.generations = kwargs.get('generations', 10000)
        self.neighborhood_size = kwargs.get('neighborhood_size', 3)

        self.population = []
        self.scalarizing_functions = []
        self.function_to_incumbents = {}

    def run(self):
        self.__generate_scalarizing_functions(self.population_size)
        self.population = self.__initialize_population(len(self.scalarizing_functions))
        self.__random_assignment()
        for i in range(self.generations):
            self.population = self.__evolve_population()

        return list(self.function_to_incumbents.values())

    def __generate_scalarizing_functions(self, n):
        profit_risk_factors = generate_risk_and_return_weights(n)
        for profit_factor, risk_factor in profit_risk_factors:
            self.scalarizing_functions.append(ScalarizingFunction(risk_factor, profit_factor))

    def __initialize_population(self, n):
        return np.array([self._generate_random_individual() for _ in range(n)])

    def _generate_random_individual(self):
        individual = np.random.rand(self.instance.assets_number)
        return individual / np.sum(individual)
        # return Individual(np.random.rand(self.instance.assets_number))

    def __random_assignment(self):
        for scalarizing_function in self.scalarizing_functions:
            idx = np.random.choice(len(self.population), replace=False)
            self.function_to_incumbents[scalarizing_function] = self.population[idx]

    def __evolve_population(self):
        scalarizing_function = list(self.function_to_incumbents.keys())
        np.random.shuffle(scalarizing_function)

        for scalarizing_function in self.scalarizing_functions:
            mating_pool = self.__create_mating_pool(scalarizing_function)
            parents = self.__select_parents(mating_pool)
            child = self.__crossover(parents)
            if scalarizing_function(child, self.instance) > scalarizing_function(
                    self.function_to_incumbents[scalarizing_function], self.instance):
                self.function_to_incumbents[scalarizing_function] = child
            # self.__update_incumbents(child, scalarizing_function, mating_pool)

    def __create_mating_pool(self, scalarizing_function):
        profit_factor, risk_factor = scalarizing_function.profit_factor, scalarizing_function.risk_factor
        most_similar_functions = sorted(self.scalarizing_functions,
                                        key=lambda f: abs(f.profit_factor - profit_factor) + abs(
                                            f.risk_factor - risk_factor))[:self.neighborhood_size]
        return [(f, self.function_to_incumbents[f]) for f in most_similar_functions]

    def __select_parents(self, mating_pool):
        np.random.shuffle(mating_pool)
        return mating_pool[:2]

    def __crossover(self, parents):
        parent1, parent2 = [parent[1] for parent in parents]
        crossover_point = np.random.randint(0, self.instance.assets_number)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child /= np.sum(child)
        return child

    def __update_incumbents(self, child, child_function, mating_pool):
        for function, incumbent in mating_pool:
            if child_function(child, self.instance) > function(incumbent, self.instance):
                self.function_to_incumbents[function] = child


def calculate_risk(weights, cov_matrix):
    return weights.dot(cov_matrix).dot(weights)


def calculate_returns(weights, predictions):
    return predictions.dot(weights)

def create_pareto_front(individuals):
    pareto_front = []
    for individual in individuals:
        risk = calculate_risk(individual, cov_matrix)
        profit = calculate_returns(individual, predictions)
        if np.sum(individual) > 1.0 + 1e-6:
            raise ValueError("Invalid weights")
        pareto_front.append((individual, profit, risk))
        pareto_front.sort(key=lambda x: x[1])
    return pareto_front


if __name__ == '__main__':
    wb = WarrenBuffett()
    wb.make_me_rich()

    predictions = np.array(list(wb.get_expected_returns().values()))
    cov_matrix = wb.get_covariance_matrix()

    # Min-Max scaling for predictions
    predictions_min = predictions.min()
    predictions_max = predictions.max()
    normalized_predictions = (predictions - predictions_min) / (predictions_max - predictions_min)

    # Min-Max scaling for covariance matrix
    cov_matrix_min = cov_matrix.min()
    cov_matrix_max = cov_matrix.max()
    normalized_cov_matrix = (cov_matrix - cov_matrix_min) / (cov_matrix_max - cov_matrix_min)

    instance = Instance(normalized_cov_matrix, normalized_predictions)
    algorithm = MOEADAlgorithm(instance, population_size=100, generations=10000, neighborhood_size=3)
    individuals = algorithm.run()

    pareto_front = create_pareto_front(individuals)
    plot_pareto_front(pareto_front)