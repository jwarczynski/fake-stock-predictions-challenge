import numpy as np
from cvxopt import matrix, solvers


class WeightedSumStrategy:
    def __init__(self):
        pass

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

    def generate_pareto_front(self, expected_returns, covariance_matrix, risk_profit_coeff, return_spread, risk_spread):
        optimal_weights_list = []
        portfolio_return_list = []

        for profit_coeff, risk_coeff in risk_profit_coeff:
            optimal_weights, portfolio_return = self.optimize_portfolio(
                expected_returns, covariance_matrix,
                profit_coeff, risk_coeff,
                return_spread, risk_spread
            )

            optimal_weights_list.append(optimal_weights)
            portfolio_return_list.append(portfolio_return)

        return optimal_weights_list, portfolio_return_list
