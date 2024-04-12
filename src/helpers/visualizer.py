import matplotlib.pyplot as plt


def plot_pareto_front(ax, solution_list, selected_index=None, title='Expected Return vs. Risk',
                      xlabel='Expected Return', ylabel='Risk'):
    """
    Plot the Pareto front.

    Parameters:
        solution_list (list): List of tuples containing solutions (weights, expected_return, risk).
        selected_index (int): Index of the selected solution in solution_list (optional).

    Example usage:
    plot_pareto_front([(weights1, expected_return1, risk1), (weights2, expected_return2, risk2), ...], selected_index=0)
    :param selected_index: index of the selected solution in solution_list
    :param solution_list:
    :param ax: pyplot axis object
    """
    expected_returns = [solution[1] for solution in solution_list]
    risks = [solution[2] for solution in solution_list]

    ax.scatter(expected_returns, risks, marker='o', color='b')
    # plt.plot(expected_returns, risks, color='r', linestyle='-', linewidth=1)

    if selected_index is not None:
        selected_solution = solution_list[selected_index]
        selected_return = selected_solution[1]
        selected_risk = selected_solution[2]
        ax.scatter(selected_return, selected_risk, color='red', marker='*', s=100)
        ax.annotate('Selected Solution', (selected_return, selected_risk),
                    xytext=(selected_return - 0.1, selected_risk + 1 * selected_risk),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


def plot_predictions(historical_data, predicted_prices, asset_name, save=False):
    last_time = len(historical_data)
    plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(historical_data, label='Real')
    plt.plot(range(last_time, last_time + len(predicted_prices)), predicted_prices, label='FFT prediction',
             linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'{asset_name} stock prices & Fourier predictions')
    plt.legend()
    if save:
        plt.savefig(f'{asset_name}_predictions.png')
    plt.show()
