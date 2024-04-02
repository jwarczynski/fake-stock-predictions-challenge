import math
import pandas as pd


class MetricEvaluator:
    def __init__(self, reference_pareto, generations_interval=100):
        self.generations_interval = generations_interval
        self.reference_pareto = reference_pareto
        self.generations = []

    def add_generation(self, generation):
        self.generations.append(generation)

    def get_dataframe(self):
        data = {
            'Generation': [],
            'Profit': [],
            'Risk': [],
            'Diversity': []
        }

        for generation_num, incumbents, returns, risks, diversity in self.generations:
            for incumbent, return_value, risk_value in zip(incumbents, returns, risks):
                data['Generation'].append(generation_num)
                data['Profit'].append(return_value)
                data['Risk'].append(risk_value)
                data['Diversity'].append(diversity)

        df = pd.DataFrame(data)
        return df

    def igd_for_generations_from_df(self, run_data):
        igd_data = pd.DataFrame(columns=['Generation', 'IGD'])
        unique_generations = run_data['Generation'].unique()
        for generation_num in unique_generations:
            generation_data = run_data[run_data['Generation'] == generation_num]

            profits = generation_data['Profit']
            risks = generation_data['Risk']

            pareto_front = [(p, r) for p, r in zip(profits, risks)]
            igd_value = self.inverted_generational_distance(pareto_front)
            igd_data = pd.concat([igd_data, pd.DataFrame({'Generation': [generation_num], 'IGD': [igd_value]})],
                                 ignore_index=True)
        return igd_data

    def igd_for_generations(self):
        igd_values = {}
        for generation in self.generations:
            generation_num = generation[0]
            profits = generation[2]
            risks = generation[3]

            pareto_front = [(p, r) for p, r in zip(profits, risks)]
            igd_values[generation[0]] = self.inverted_generational_distance(pareto_front)
        return igd_values

    def inverted_generational_distance(self, pareto):
        reference_pareto = self.__normalize_pareto(self.reference_pareto)
        pareto = self.__normalize_pareto(pareto)

        total_distance = 0.0
        for s in reference_pareto:
            nearest_neighbor_distance = min(self.__euclidean_distance(s, p) for p in pareto)**2
            total_distance += nearest_neighbor_distance

        igd = math.sqrt(total_distance) / len(reference_pareto)
        return igd

    def __normalize_pareto(self, pareto):
        max_values = [max(pareto, key=lambda x: x[i])[i] for i in range(len(pareto[0]))]
        # Normalize each value in each tuple
        return [[x / max_values[i] for i, x in enumerate(point)] for point in pareto]

    def __euclidean_distance(self, point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    @staticmethod
    def calculate_hv(population, pareto_front):
        return 0
