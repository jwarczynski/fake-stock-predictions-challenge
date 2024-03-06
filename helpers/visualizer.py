import matplotlib.pyplot as plt


def plot_pareto_front(solution_list):
    """
    Example usage:
    solution_list = [(weights1, expected_return1, risk1), (weights2, expected_return2, risk2), ...]
    """
    expected_returns = [solution[1] for solution in solution_list]
    risks = [solution[2] for solution in solution_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(expected_returns, risks, marker='o', color='b')
    plt.plot(expected_returns, risks, color='r', linestyle='-', linewidth=1)

    plt.xlabel('Expected Return')
    plt.ylabel('Risk')
    plt.title('Expected Return vs. Risk')
    plt.grid(True)
    plt.show()


def plot_predictions(historical_data, predicted_prices, asset_name):
    last_time = len(historical_data)
    plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(historical_data, label='Real')
    plt.plot(range(last_time, last_time + len(predicted_prices)), predicted_prices, label='FFT prediction', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'{asset_name} stock prices & Fourier predictions')
    plt.legend()
    plt.show()
