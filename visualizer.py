import matplotlib.pyplot as plt


def plot_pareto_front(solution_list):
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

# Example usage
# solution_list = [(weights1, expected_return1, risk1), (weights2, expected_return2, risk2), ...]

# Call the function with your solution list
# plot_expected_return_vs_risk(solution_list)
