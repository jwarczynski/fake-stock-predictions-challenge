import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from helpers.data_loader import load_data
from fortune_tellers.fft_fortune_teller import FFTFortuneTeller
from helpers.visualizer import plot_predictions
from startegies.weighted_sum_strategy import WeightedSumStrategy
from startegies.epsilon_constrained_strategy import EpsilonConstrainedStrategy

class WarrenBuffett:
    def __init__(self, strategy="wsm", path="Bundle2", extension=".txt"):
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

    def get_expected_returns(self):
        return self.__asset_expected_returns

    def get_covariance_matrix(self):
        return self.__covariance_matrix

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
            last_price = measurements[-1][1]
            future_price_prediction = self.__asset_predictions[asset_name][-1]
            expected_return = (future_price_prediction - last_price) / last_price
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

    import matplotlib.colors as mcolors

    def plot_all_predictions(self):
        num_assets = len(self.__asset_predictions)
        num_rows = 5
        num_cols = (num_assets + num_rows - 1) // num_rows  # Calculate number of columns dynamically
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 30))

        asset_names = list(self.__asset_predictions.keys())
        num_bundles = (len(self.__asset_data) + num_rows - 1) // num_rows
        base_hue = 0.5
        hue_increment = 0.4 / num_bundles  # Adjust the increment for subtle hue differences

        for i, asset_name in enumerate(asset_names):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_assets > 1 else axes  # Handle the case when there is only one asset

            # Plot historical data
            historical_data = [price for _, price in self.__asset_data[asset_name]]
            x = list(range(len(historical_data)))
            bundle_index = 0
            for j in range(1, len(x), 100):
                hue = (base_hue + bundle_index * hue_increment) % 1.0
                bundle_color = mcolors.hsv_to_rgb((hue, 0.8, 0.8))  # Adjust saturation and value for clarity
                ax.plot(x[j:j + 100], historical_data[j:j + 100], color=bundle_color,
                        label=f"{asset_name} (Historical)", linestyle='-' if j == 0 else '--')
                bundle_index += 1

            # Plot predictions
            predicted_prices = self.__asset_predictions[asset_name]
            ax.plot(range(len(historical_data), len(historical_data) + len(predicted_prices)), predicted_prices,
                    color='orange', label=f"{asset_name} (Predictions)")

            ax.set_title(asset_name)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, historical_data, predicted_prices, asset_name, ax):
        last_time = len(historical_data)
        ax.plot(historical_data, label='Real')
        ax.plot(range(last_time, last_time + len(predicted_prices)), predicted_prices, label='FFT prediction',
                linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(f'{asset_name} stock prices & Fourier predictions')
        ax.legend()

    def get_assets_keys(self):
        return np.array(list(self.__asset_data.keys()))

    def show_predictions_for_asset(self, asset_names):
        for asset_name in asset_names:
            historical_data = [price for _, price in self.__asset_data[asset_name]]
            predicted_prices = self.__asset_predictions[asset_name]
            plot_predictions(historical_data, predicted_prices, asset_name)

    def show_weights_for_assets(self, asset_weights):
        asset_names = list(asset_weights.keys())
        weights = list(asset_weights.values())

        plt.figure(figsize=(10, 6))
        plt.barh(asset_names, [w * 100 for w in weights], color='skyblue')  # Multiply weights by 100 to show as percentage
        plt.xlabel('Percentage of Total Budget')
        plt.ylabel('Asset Name')
        plt.title('Portfolio Allocation')
        plt.gca().invert_yaxis()  # Invert y-axis to display top-down
        plt.tight_layout()  # Adjust layout to prevent cropping
        plt.show()
