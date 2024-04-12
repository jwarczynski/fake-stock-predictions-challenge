import matplotlib.pyplot as plt
import seaborn as sns


def plot_pareto_fronts(ax, pareto_fronts, linear_front_index=None):
    # Define a color palette for seaborn
    palette = sns.color_palette("husl", len(pareto_fronts))
    for i, pareto_front in enumerate(pareto_fronts):
        profits = [point[0] for point in pareto_front]
        risks = [point[1] for point in pareto_front]
        if i == linear_front_index:
            sns.scatterplot(x=profits, y=risks, label=f'Pareto Front {i+1}', marker='o', color=palette[i])
        else:
            if i == 1:  # Increase marker size for Pareto front 2
                sns.scatterplot(x=profits, y=risks, label=f'Pareto Front {i+1}', color=palette[i], s=100)
            else:
                sns.scatterplot(x=profits, y=risks, label=f'Pareto Front {i+1}', color=palette[i], s=50)
    ax.set_xlabel('Profit')
    ax.set_ylabel('Risk')
    ax.set_title('Pareto Fronts')
    ax.legend()


def get_pareto_fronts_plot(pareto_fronts, ax=None, algorithm_params=None, reference_pareto=None, labels=None, ref_pareto_label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Pareto fronts
    palette = sns.color_palette("husl", len(pareto_fronts) + 1)
    for i, pareto_front in enumerate(pareto_fronts):
        profits = [point[0] for point in pareto_front]
        risks = [point[1] for point in pareto_front]
        ax.scatter(profits, risks, label=f'Pareto Front {i + 1}' if labels is None else labels[i], color=palette[i])

    # Plot reference Pareto front
    if reference_pareto is not None:
        ref_profits = [point[0] for point in reference_pareto]
        ref_risks = [point[1] for point in reference_pareto]
        ax.plot(ref_profits, ref_risks, label='Reference Pareto Front' if ref_pareto_label is None else ref_pareto_label, color=palette[len(pareto_fronts)])

        # # Add a specific legend entry for the reference Pareto front
        # if algorithm_params is not None:
        #     ax.legend([f'{key}: {val}' for key, val in algorithm_params.items()] + ['Reference Pareto Front'], loc='upper left')
        # else:
        #     ax.legend(['Reference Pareto Front'], loc='upper left')

    else:
        if algorithm_params is not None:
            ax.legend([f'{key}: {val}' for key, val in algorithm_params.items()], loc='upper left')
        else:
            ax.legend()

    ax.legend()

    # ax.set_xlabel(labels[0] if labels is not None else 'Profit')
    # ax.set_ylabel(labels[1] if labels is not None else 'Risk')

    ax.set_title('Pareto Fronts')

    if ax is None:
        return fig, ax
    else:
        return ax
