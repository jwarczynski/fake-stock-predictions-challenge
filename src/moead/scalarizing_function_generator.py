from .scalarizing_function import ScalarizingFunction


class ScalarizingFunctionGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_functions(num_objectives=2, n=100):
        if num_objectives == 2:
            return [ScalarizingFunction(w1, w2, 0) for w1, w2 in ScalarizingFunctionGenerator.generate_risk_and_return_weights(n)]
        elif num_objectives == 3:
            print("3 objectives")
            return [ScalarizingFunction(f1, f2, f3) for f1, f2, f3 in ScalarizingFunctionGenerator.generate_weights_3_objectives(n)]
        else:
            raise ValueError("Number of objectives must be 2 or 3.")

    @staticmethod
    def generate_risk_and_return_weights(n=100):
        weights = []
        for j1 in range(n):
            return_weight = 1 / n * j1
            risk_weight = 1 - return_weight
            weights.append((return_weight, risk_weight))
        return weights

    @staticmethod
    def generate_weights_3_objectives(n=100):
        weights = []
        for j1 in range(n + 1):  # Adjust range to include 0 and n
            for j2 in range(n + 1 - j1):  # Adjust range based on j1
                j3 = n - j1 - j2
                k1 = j1 / n
                k2 = j2 / n
                k3 = j3 / n
                weights.append((k1, k2, k3))

        return weights[::len(weights) // n]
