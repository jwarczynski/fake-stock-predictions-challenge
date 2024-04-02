from src.moead.metrics.evaluator import MetricEvaluator


def calculate_igd_for_group(group, reference_pareto):
    metric_evaluator = MetricEvaluator(reference_pareto, generations_interval=100)
    pareto = list(zip(group['Profit'], group['Risk']))
    return metric_evaluator.inverted_generational_distance(pareto)