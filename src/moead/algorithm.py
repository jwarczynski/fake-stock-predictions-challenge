from typing import List

import numpy as np

from .instance import Instance
from .scalarizing_function import ScalarizingFunction
from .utils import calculate_returns, calculate_risk


class MOEADAlgorithm:
    def __init__(self, instance: Instance, scalarizing_functions: List[ScalarizingFunction], **kwargs):
        self.original_instance = instance
        self.instance = instance.normalize()

        self.population_size = kwargs.get('population_size', len(scalarizing_functions))
        self.generations = kwargs.get('generations', 10000)
        self.neighborhood_size = kwargs.get('neighborhood_size', 5)

        self.scalarizing_functions = scalarizing_functions
        self.function_to_incumbents = {}
        self.population = []

    def run(self, metric_evaluator=None):
        self.population = self.__initialize_population(self.population_size)
        self.__random_assignment()
        # self.__greedy_assignment()
        for i in range(0, self.generations + 1):
            if metric_evaluator is not None and i % metric_evaluator.generations_interval == 0:
                metric_evaluator.add_generation(
                    (
                        i,
                        list(self.function_to_incumbents.values()),
                        [calculate_returns(incumbent, self.original_instance.predictions) for incumbent in list(self.function_to_incumbents.values())],
                        [calculate_risk(incumbent, self.original_instance.cov_matrix) for incumbent in list(self.function_to_incumbents.values())],
                        [np.sum(np.where(incumbent < 1e-2, 1, 0)) for incumbent in list(self.function_to_incumbents.values())]
                    )
                )
            self.__evolve_population()
            self.mutate_population()

        return list(self.function_to_incumbents.values())

    def __initialize_population(self, n):
        high_risk_vectors =  self.generate_vectors(n)
        # return high_risk_vectors
        random_vectors = np.array([self._generate_random_individual() for _ in range(n)])
        return np.concatenate((high_risk_vectors, random_vectors))
        # pop = np.array([self._generate_random_individual() for _ in range(n)])
        # for i in range(self.instance.assets_number):
        #     individual = np.zeros(self.instance.assets_number)
        #     individual[i] = 1
        #     np.append(pop, individual)
        # return pop
        # return np.array([self._generate_random_individual() for _ in range(n-20)])

    def _generate_random_individual(self):
        random_numbers = np.random.rand(self.instance.assets_number - 1)
        random_numbers.sort()
        return np.diff(np.concatenate(([0], random_numbers, [1])))

    def __random_assignment(self):
        for scalarizing_function in self.scalarizing_functions:
            idx = np.random.choice(len(self.population), replace=False)
            self.function_to_incumbents[scalarizing_function] = self.population[idx]

    def __greedy_assignment(self):
        for scalarizing_function in self.scalarizing_functions:
            self.function_to_incumbents[scalarizing_function] = max(self.population, key=lambda incumbent: scalarizing_function(incumbent, self.instance))


    def generate_vectors(self, num_vectors=100, max_nonzero=2):
        vectors = []
        for _ in range(num_vectors):
            # Generate random indices for nonzero weights
            indices = np.random.choice(20, np.random.randint(1, max_nonzero + 1), replace=False)
            # Generate random weights for the selected indices
            weights = np.zeros(20)
            for idx in indices:
                weights[idx] = np.random.uniform(0, 1)
            # Normalize the weights to ensure they sum up to 1
            weights /= np.sum(weights)
            vectors.append(weights)
        return np.array(vectors)

    def __evolve_population(self):
        scalarizing_functions = list(self.function_to_incumbents.keys())
        np.random.shuffle(scalarizing_functions)

        for scalarizing_function in scalarizing_functions:
            mating_pool = self.__create_mating_pool(scalarizing_function)
            parents = self.__select_parents(mating_pool)
            child = self.__crossover(parents)
            if scalarizing_function(child, self.instance) > scalarizing_function(
                    self.function_to_incumbents[scalarizing_function], self.instance):
                self.function_to_incumbents[scalarizing_function] = child
            # self.__update_incumbents(child, mating_pool)

    def __create_mating_pool(self, scalarizing_function):
        profit_factor, risk_factor, divers_factor = scalarizing_function.profit_factor, scalarizing_function.risk_factor, scalarizing_function.diversification_factor
        most_similar_functions = sorted(self.scalarizing_functions,
                                        key=lambda f: abs(f.profit_factor - profit_factor) + abs(
                                            f.risk_factor - risk_factor) + abs(f.diversification_factor - divers_factor))[:self.neighborhood_size]
        return [(f, self.function_to_incumbents[f]) for f in most_similar_functions]

    def __select_parents(self, mating_pool):
        np.random.shuffle(mating_pool)
        return mating_pool[:3]

    def __crossover(self, parents):
        parent1, parent2, parent3 = [parent[1] for parent in parents]
        return self.vector_difference_crossover(parent1, parent2)
        scaling_factor = np.random.rand()
        child = parent1 + (parent3 - parent2) * scaling_factor
        # child = np.clip(child, 0, 1)
        # counter = 0
        # while np.sum(child) < 0.99999 or np.sum(child) > 1.000001:
        #     over = np.sum(child) - 1
        #     over_per_element = over / len(child)
        #     child = child - over_per_element
        #     child = np.clip(child, 0, 1)
        #     counter += 1
        #     if counter > 10:
        #         raise ValueError("Invalid child")
        return child

    def vector_difference_crossover(self, parent1, parent2):
        """
        Vector Difference Crossover (VDC) operator for MOEAD genetic algorithm.

        Args:
        - parent1 (ndarray): First parent vector
        - parent2 (ndarray): Second parent vector

        Returns:
        - child (ndarray): Offspring vector
        """
        scaling_factor = np.random.rand()
        child = parent1 + (parent2 - parent1) * scaling_factor
        child = np.clip(child, 0, 1)  # Clip values to ensure feasibility
        return child / np.sum(child)  # Normalize to ensure sum equals 1

    def blend_crossover(self, parent1, parent2, alpha=0.1):
        """
        Blend Crossover (BLX-α) operator for MOEAD genetic algorithm.

        Args:
        - parent1 (ndarray): First parent vector
        - parent2 (ndarray): Second parent vector
        - alpha (float): Blend parameter

        Returns:
        - child (ndarray): Offspring vector
        """
        min_values = np.minimum(parent1, parent2)
        max_values = np.maximum(parent1, parent2)
        range_values = max_values - min_values
        lower_bound = min_values - alpha * range_values
        upper_bound = max_values + alpha * range_values
        child = np.random.uniform(lower_bound, upper_bound)
        # child = np.clip(child, 0, 1)  # Clip values to ensure feasibility
        # if np.sum(child) < 0.99999 or np.sum(child) > 1.000001:
        #     child = child/np.sum(child)
        # Adjust the child vector to ensure sum equals 1
        # child /= np.sum(child)  # Normalize to ensure sum equals 1
        # excess = np.sum(child) - 1
        # child -= excess / len(child)  # Adjust to ensure sum equals 1
        return child

    def __update_incumbents(self, child, mating_pool):
        for function, incumbent in mating_pool:
            if function(child, self.instance) > function(incumbent, self.instance):
                self.function_to_incumbents[function] = child

    def mutate_population(self):
        for scalarizing_function in self.scalarizing_functions:
            incumbent = self.function_to_incumbents[scalarizing_function]
            # if np.random.rand() < 0.05:
            # self.function_to_incumbents[scalarizing_function] = self.gaussian_mutation(incumbent)
            #     self.function_to_incumbents[scalarizing_function] = self.custom_mutation(incumbent)
            # self.function_to_incumbents[scalarizing_function] = self.gaussian_mutation(incumbent)
            if np.random.rand() < 0.05:
                self.function_to_incumbents[scalarizing_function] = self.custom_mutation(incumbent)

    def gaussian_mutation(self, individual, mutation_rate=0.1, sigma=0.1):
        """
        Gaussian Mutation operator for MOEAD genetic algorithm.

        Args:
        - individual (ndarray): Individual vector
        - mutation_rate (float): Probability of mutation for each element
        - sigma (float): Standard deviation for Gaussian mutation

        Returns:
        - mutated_individual (ndarray): Mutated individual vector
        """
        mutated_individual = np.copy(individual)
        for i in range(len(mutated_individual)):
            if np.random.rand() < mutation_rate:
                mutated_individual[i] += np.random.normal(0, sigma)
                mutated_individual[i] = max(0, mutated_individual[i])  # Ensure non-negativity
        mutated_individual /= np.sum(mutated_individual)  # Normalize to ensure sum equals 1
        return mutated_individual

    def custom_mutation(self, individual, sigma=0.1):
        """
        Custom Mutation operator for MOEAD genetic algorithm.

        Args:
        - individual (ndarray): Individual vector
        - sigma (float): Standard deviation for mutation

        Returns:
        - mutated_individual (ndarray): Mutated individual vector
        """
        mutated_individual = np.copy(individual)
        # Choose two different indices randomly
        index1, index2 = np.random.choice(len(mutated_individual), 2, replace=False)
        max_amount = min(mutated_individual[index1], 1 - mutated_individual[index2])
        mutation_amount = np.random.uniform(0, max_amount)
        # Determine the maximum amount that can be added/subtracted to keep the weight within [0, 1]
        # max_amount_index1 = min(mutated_individual[index1], 1 - mutated_individual[index2])
        # max_amount_index2 = min(mutated_individual[index2], 1 - mutated_individual[index1])
        # # Randomly select an amount within the valid range
        # mutation_amount = np.random.uniform(-max_amount_index1, max_amount_index2)
        # Add the amount to index1 and subtract from index2
        mutated_individual[index1] -= mutation_amount
        mutated_individual[index2] += mutation_amount
        # Ensure the mutated weights remain within the range [0, 1]
        # mutated_individual[index1] = max(0, min(1, mutated_individual[index1]))
        # mutated_individual[index2] = max(0, min(1, mutated_individual[index2]))
        return mutated_individual

    def swap_indices_mutation(self, individual):
        """
        Swap Indices Mutation operator for MOEAD genetic algorithm.

        Args:
        - individual (ndarray): Individual vector

        Returns:
        - mutated_individual (ndarray): Mutated individual vector
        """
        mutated_individual = np.copy(individual)
        # Choose two different indices randomly
        index1, index2 = np.random.choice(len(mutated_individual), 2, replace=False)
        # Swap the weights at the selected indices
        mutated_individual[index1], mutated_individual[index2] = mutated_individual[index2], mutated_individual[index1]
        return mutated_individual
