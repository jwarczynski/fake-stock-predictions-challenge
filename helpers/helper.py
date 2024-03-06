def calculate_risk(weights, covariance_matrix):
    """
    Calculate the risk of a portfolio.

    Parameters:
        - weights (array): Portfolio weights.
        - covariance_matrix (array): Covariance matrix.

    Returns:
        - float: Portfolio risk.
    """
    s = 0
    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            s += w1 * w2 * covariance_matrix[i, j]
    return s
    # risk = np.dot(weights.T, np.dot(covariance_matrix, weights))
    # return np.sqrt(risk)


def generate_risk_and_return_weights(n=100):
    """
    Generate k1 and k2 values based on a specific pattern.

    Parameters:
        - M (int): Number of objectives.
        - n (int): Number of times a dimension is split.

    Returns:
        - List of tuples: Each tuple contains k1 and k2 values.
    """
    weights = []
    for j1 in range(n + 1):
        return_weight = 1 / n * j1
        risk_weight = 1 - return_weight
        weights.append((return_weight, risk_weight))
    return weights
