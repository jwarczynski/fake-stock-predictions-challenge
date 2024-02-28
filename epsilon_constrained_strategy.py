import numpy as np
from cvxopt import matrix, solvers


class EpsilonConstrainedStrategy:
    def __init__(self):
        pass

    def optimize_portfolio(self, expected_returns, covariance_matrix, target_return, _=None, __=None, ___=None):
        n = len(expected_returns)

        # Quadratic term in the objective function
        Q = matrix(2 * covariance_matrix)

        # Linear term in the objective function
        c = matrix(0.0, (n, 1))

        # Constraint: expected return >= target return
        G = matrix(-expected_returns, (1, n))
        h = matrix(-target_return)

        # Constraint: sum of weights = 1
        Aeq = matrix(1.0, (1, n))
        beq = matrix(1.0)

        sol = solvers.qp(Q, c, G, h, Aeq, beq)

        optimal_weights = np.array(sol['x']).flatten()
        portfolio_risk = sol['primal objective']

        return optimal_weights, portfolio_risk

    def generate_pareto_front(self, expected_returns, covariance_matrix):
        expected_return_constraints = np.linspace(
            max(0, min(expected_returns)), max(expected_returns), 20
        )

        optimal_weights_list = []
        portfolio_risk_list = []

        for return_constraint in expected_return_constraints:
            optimal_weights, risk = self.optimize_portfolio(
                expected_returns, covariance_matrix, return_constraint
            )

            optimal_weights_list.append(optimal_weights)
            portfolio_risk_list.append(risk)

        return optimal_weights_list, expected_return_constraints, portfolio_risk_list
