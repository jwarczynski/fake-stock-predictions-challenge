import numpy as np
from cvxopt import matrix, solvers

from helper import generate_risk_and_return_weights, calculate_risk


class WeightedSumStrategy:
    def __init__(self):
        pass

    def __calculate_return_and_risk_spreads(self, asset_expected_returns, covariance_matrix):
        expected_returns = np.array(list(asset_expected_returns))
        max_profit_weights, max_returns = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            1, 0
        )
        min_risk_weights, min_risk_returns = self.optimize_portfolio(
            expected_returns, covariance_matrix,
            0, 1
        )

        max_risk = calculate_risk(max_profit_weights, covariance_matrix)
        min_risk = calculate_risk(min_risk_weights, covariance_matrix)

        return max_returns - min_risk_returns, max_risk - min_risk

    def optimize_portfolio(
            self, expected_returns, covariance_matrix, profit_weight,
            risk_weight, return_spread=1.0, risk_spread=1.0
    ):
        n = len(expected_returns)

        c = matrix(-profit_weight / return_spread * np.array(expected_returns))  # Vector of expected returns
        Sigma = matrix(risk_weight / risk_spread * np.array(covariance_matrix))  # Covariance matrix
        Q = 2 * Sigma  # Quadratic term in the objective function

        # Equality constraint: sum of weights = 1
        A = matrix(1.0, (1, n))
        b = matrix(1.0)

        # Inequality constraints: weights >= 0
        G = matrix(-np.eye(n))
        h = matrix(0.0, (n, 1))

        sol = solvers.qp(Q, c, G, h, A, b)

        optimal_weights = np.array(sol['x'])
        portfolio_return = -sol['primal objective']

        return optimal_weights, portfolio_return

    def generate_pareto_front(self, expected_returns, covariance_matrix):
        risk_profit_coeff = generate_risk_and_return_weights()
        return_spread, risk_spread = self.__calculate_return_and_risk_spreads(expected_returns, covariance_matrix)

        optimal_weights_list = []
        portfolio_return_list = []
        risks_list = []

        for profit_coeff, risk_coeff in risk_profit_coeff:
            optimal_weights, portfolio_return = self.optimize_portfolio(
                expected_returns, covariance_matrix,
                profit_coeff, risk_coeff,
                return_spread, risk_spread
            )
            risk = calculate_risk(optimal_weights, covariance_matrix)

            optimal_weights_list.append(optimal_weights)
            portfolio_return_list.append(portfolio_return)
            risks_list.append(risk)

        return optimal_weights_list, portfolio_return_list, risks_list
