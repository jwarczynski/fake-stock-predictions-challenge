from .evaluator import MetricEvaluator
import os


class HistoryWriter:
    def __init__(self, directory):
        self.directory = directory

    def dump(self, metric_evaluator: MetricEvaluator, run_id):
        for generation in metric_evaluator.generations:
            directory_path = f"{self.directory}/run_{run_id}"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            with open(f"{directory_path}/generation_{generation[0]}.csv", 'w') as f:
                f.write("Profit, Risk, Diversity\n")

                incumbents = generation[1]
                profits = generation[2]
                risks = generation[3]
                diversities = generation[4]
                for incumbent, profit, risk, diversity in zip(incumbents, profits, risks, diversities):
                    f.write("{}, {}, {}\n".format(profit, risk, diversity))
