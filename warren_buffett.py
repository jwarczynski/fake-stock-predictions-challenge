import numpy as np
from sklearn.linear_model import LinearRegression

from data_loader import load_data
from helper import calculate_risk, generate_risk_and_return_weights
from weighted_sum_strategy import WeightedSumStrategy


class WarrenBuffett:
    def __init__(self, path="Bundle1", extension=".txt", strategy="wcm"):
        self.__asset_data = load_data(path, extension)

        self.__asset_predictions = {}
        self.__asset_expected_returns = {}
        self.__asset_std_devs = {}
        self.__covariance_matrix = None

        self.__returns_spread = None
        self.__risk_spread = None

        self.__investment_profiles = []

        if strategy == "wcm":
            self.strategy = WeightedSumStrategy()
        elif strategy == "ecm":
            pass
        else:
            raise ValueError("Invalid strategy")

    def explain(self, method="wcm"):
        pass

    def make_me_rich(self):
        self.__make_predictions()
        self.__calculate_expected_returns()
        self.__calculate_std_devs()
        self.__calculate_covariance_matrix()
        self.__generate_pareto_front()

        return self.get_investment_profiles()

    def set_strategy(self, strategy):
        if strategy == "wcm":
            self.strategy = WeightedSumStrategy()
        elif strategy == "ecm":
            pass
        else:
            raise ValueError("Invalid strategy")

    def __make_predictions(self):
        for asset_name, measurements in self.__asset_data.items():
            times = [point[0] for point in measurements]
            prices = [point[1] for point in measurements]
            X = np.array(times).reshape(-1, 1)
            y = np.array(prices)
            model = LinearRegression()
            model.fit(X, y)

            # Predict stock prices for time 101 to 200
            future_times = np.arange(101, 201).reshape(-1, 1)
            predicted_prices = model.predict(future_times)
            self.__asset_predictions[asset_name] = predicted_prices

    def __calculate_expected_returns(self):
        for asset_name, measurements in self.__asset_data.items():
            price_t100 = next((price for time, price in measurements if time == 100), None)
            predicted_price_t200 = self.__asset_predictions[asset_name][-1]

            if price_t100 is not None:
                expected_return = (predicted_price_t200 - price_t100) / price_t100
            else:
                expected_return = None

            self.__asset_expected_returns[asset_name] = expected_return

    def __calculate_std_devs(self, window_size=10):
        for asset_name, measurements in self.__asset_data.items():
            prices = [point[1] for point in measurements]
            std_devs = np.zeros(len(prices) - window_size + 1)

            for i in range(len(prices) - window_size + 1):
                window = prices[i:i + window_size]
                std_devs[i] = np.std(window)

            self.__asset_std_devs[asset_name] = std_devs

    def __calculate_covariance_matrix(self):
        num_assets = len(self.__asset_std_devs)
        self.__covariance_matrix = np.zeros((num_assets, num_assets))

        for i, (asset1, std_dev1) in enumerate(self.__asset_std_devs.items()):
            for j, (asset2, std_dev2) in enumerate(self.__asset_std_devs.items()):
                self.__covariance_matrix[i, j] = np.cov(std_dev1, std_dev2)[0, 1]

    def __calculate_return_and_risk_spreads(self):
        expected_returns = np.array(list(self.__asset_expected_returns.values()))
        max_profit_weights, max_returns = self.strategy.optimize_portfolio(
            expected_returns, self.__covariance_matrix,
            1, 0
        )
        min_risk_weights, min_risk_returns = self.strategy.optimize_portfolio(
                expected_returns, self.__covariance_matrix,
                0, 1
            )

        max_risk = calculate_risk(max_profit_weights, self.__covariance_matrix)
        min_risk = calculate_risk(min_risk_weights, self.__covariance_matrix)

        self.__returns_spread = max_returns - min_risk_returns
        self.__risk_spread = max_risk - min_risk

        return max_returns - min_risk_returns, max_risk - min_risk

    def __generate_pareto_front(self):
        self.__calculate_return_and_risk_spreads()
        coefficients = generate_risk_and_return_weights()

        expected_returns = np.array(list(self.__asset_expected_returns.values()))
        pareto_solutions = self.strategy.generate_pareto_front(
            expected_returns, self.__covariance_matrix, coefficients,
            self.__returns_spread, self.__risk_spread
        )

        for weights, gain in zip(*pareto_solutions):
            risk = calculate_risk(weights, self.__covariance_matrix)
            self.__investment_profiles.append((weights, gain, risk))

    def get_investment_profiles(self):
        return self.__investment_profiles
