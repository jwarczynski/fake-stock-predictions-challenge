from .instance import Instance
import numpy as np


class ScalarizingFunction:
    def __init__(self, risk_factor, profit_factor, diversification_factor=0.0):
        self.risk_factor = risk_factor
        self.profit_factor = profit_factor
        self.diversification_factor = diversification_factor

    def __call__(self, individual, instance: Instance):
        return self.profit_factor * instance.predictions.dot(individual) - self.risk_factor * individual.dot(
            instance.cov_matrix).dot(individual) - self.diversification_factor * np.sum(
            np.where(individual < 1e-2, 1, 0) / instance.assets_number)

    def __eq__(self, other):
        if isinstance(other, ScalarizingFunction):
            return (self.risk_factor, self.profit_factor, self.diversification_factor) == (
            other.risk_factor, other.profit_factor, self.diversification_factor)
        return False

    def __hash__(self):
        return hash((self.risk_factor, self.profit_factor, self.diversification_factor))
