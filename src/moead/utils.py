import numpy as np


def calculate_risk(weights, cov_matrix):
    return weights.dot(cov_matrix).dot(weights)


def calculate_returns(weights, predictions):
    return predictions.dot(weights)


def create_pareto_front(individuals, predictions, cov_matrix, num_objectives=2):
    pareto_front = []
    for individual in individuals:
        risk = calculate_risk(individual, cov_matrix)
        profit = calculate_returns(individual, predictions)
        if np.sum(individual) > 1.0 + 1e-6:
            raise ValueError("Invalid weights")
        if num_objectives == 3:
            diversification = np.sum(np.where(individual < 1e-2, 1, 0))
            pareto_front.append((individual, profit, risk, (20 - diversification)/20))
        else:
            pareto_front.append((individual, profit, risk))
    return pareto_front
