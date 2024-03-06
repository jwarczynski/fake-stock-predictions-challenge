import numpy as np

from helpers.data_loader import load_data
from fortune_tellers.fft_fortune_teller import FFTFortuneTeller
from helpers.visualizer import plot_predictions
from startegies.weighted_sum_strategy import WeightedSumStrategy
from startegies.epsilon_constrained_strategy import EpsilonConstrainedStrategy

class WarrenBuffett:
    def __init__(self, strategy="wsm", path="Bundle1", extension=".txt"):
        self.__asset_data = load_data(path, extension)

        self.__asset_predictions = {}
        self.__asset_expected_returns = {}
        self.__asset_std_devs = {}
        self.__covariance_matrix = None

        self.__returns_spread = None
        self.__risk_spread = None

        self.__investment_profiles = []

        self.fortune_teller = FFTFortuneTeller()

        if strategy == "wsm":
            self.strategy = WeightedSumStrategy()
        elif strategy == "ecm":
            self.strategy = EpsilonConstrainedStrategy()
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
        if strategy == "wsm":
            self.strategy = WeightedSumStrategy()
        elif strategy == "ecm":
            pass
        else:
            raise ValueError("Invalid strategy")

    def __make_predictions(self):
        for asset_name, measurements in self.__asset_data.items():
            times = np.array([point[0] for point in measurements])
            prices = np.array([point[1] for point in measurements])
            predicted_prices = self.fortune_teller.make_prediction(times, prices)
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

    def __generate_pareto_front(self):
        expected_returns = np.array(list(self.__asset_expected_returns.values()))
        pareto_solutions = self.strategy.generate_pareto_front(
            expected_returns, self.__covariance_matrix,
        )

        for weights, gain, risk in zip(*pareto_solutions):
            self.__investment_profiles.append((weights, gain, risk))

    def get_investment_profiles(self):
        return self.__investment_profiles

    def plot_all_predictions(self):
        for asset_name, predicted_prices in self.__asset_predictions.items():
            historical_data = [price for _, price in self.__asset_data[asset_name]]
            plot_predictions(historical_data, predicted_prices, asset_name)

    def get_assets_keys(self):
        return np.array(list(self.__asset_data.keys()))

    def show_predictions_for_asset(self, asset_names):
        for asset_name in asset_names:
            historical_data = [price for _, price in self.__asset_data[asset_name]]
            predicted_prices = self.__asset_predictions[asset_name]
            plot_predictions(historical_data, predicted_prices, asset_name)
