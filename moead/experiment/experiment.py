from tqdm import tqdm

from moead.algorithm import MOEADAlgorithm
from moead.instance import Instance
from moead.scalarizing_function_generator import ScalarizingFunctionGenerator
from moead.metrics.evaluator import MetricEvaluator
from moead.metrics.writer import HistoryWriter


def run_experiment(instance: Instance, function_numbers, generations, run_per_config, reference_pareto):
    total = len(function_numbers) * len(generations) * run_per_config

    with tqdm(total=total) as pbar:
        for f_num in function_numbers:
            for gen in generations:
                writer = HistoryWriter(f"history3/MOEAD_{f_num}_{gen}")
                for i in range(run_per_config):
                    algorithm = MOEADAlgorithm(instance, ScalarizingFunctionGenerator.generate_functions(num_objectives=2, n=f_num), population_size=f_num*2, generations=gen, neighborhood_size=3)
                    metric_evaluator = MetricEvaluator(reference_pareto, generations_interval=gen//10)
                    individuals = algorithm.run(metric_evaluator)
                    writer.dump(metric_evaluator, i)

                    pbar.set_description(f"Function Number: {f_num}, Gen: {gen} Run: {i}")
                    pbar.update(1)