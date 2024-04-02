import matplotlib.pyplot as plt


def get_generations_plot(all_data, ax=None, num_scalar_funcs=None, ylim=None, xlim=None, xlabel='Profit', ylabel='Risk',
                         legend=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if ylim is not None:
        ax.set_ylim(ylim)
    # else:
    #     ax.set_ylim((0, all_data[all_data['Risk'] < 0.1]['Risk'].max() * 1.1))

    if xlim is not None:
        ax.set_xlim(xlim)

    sc = ax.scatter(x=all_data['Profit'], y=all_data['Risk'], c=all_data['Generation'], cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax, label='Generation Number', ticks=all_data['Generation'].unique()[::100].astype(int))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add custom legend label
    if num_scalar_funcs is not None:
        ax.legend(['Number of Scalarizing Functions: {}'.format(num_scalar_funcs)], loc='upper left')
    else:
        ax.get_legend().remove()

    if ax is None:
        return fig, ax
    else:
        return ax


def get_igd_for_generations_plot(igd_values, ax=None, num_scalar_funcs=None, title="IGD for Generations"):
    if ax is None:
        fig, ax = plt.subplots()

    # Extracting generations and corresponding IGD values
    generations = list(igd_values.keys())
    igd_scores = list(igd_values.values())

    # Plotting
    ax.scatter(generations, igd_scores, marker='o', linestyle='-')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Inverted Generational Distance (IGD)')
    ax.set_title(title)
    ax.grid(True)

    # Add custom legend labels for algorithm parameters
    if num_scalar_funcs is not None:
        ax.legend(['Number of Scalarizing Functions: {}'.format(num_scalar_funcs)], loc='upper right')

    if ax is None:
        return fig, ax
    else:
        return ax


def plot_igd_for_generations(ax, igd_values, title=None):
    # Extracting generations and corresponding IGD values
    generations = list(igd_values.keys())
    igd_scores = list(igd_values.values())

    ax.scatter(generations, igd_scores, marker='o', linestyle='-')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Inverted Generational Distance (IGD)')
    ax.set_title(title)
    ax.grid(True)


def show_all_generation_plots(df):
    fig, axs = plt.subplots(df['TotalGenerations'].nunique(), df['Scalar_funcs'].nunique(), figsize=(15, 10))
    for i, gen in enumerate(sorted(df['TotalGenerations'].unique())):
        for j, scalar_funcs in enumerate(sorted(df['Scalar_funcs'].unique())):
            data = df[(df['TotalGenerations'] == gen) & (df['Scalar_funcs'] == scalar_funcs)]
            ax = axs[i, j]
            ax.set_xlabel(None)
            if j == 0:
                ylabel = f"Generations: {gen}"
            else:
                ylabel = None

            if i == 0:
                title = f"Scalarizing Functions: {scalar_funcs}"
            else:
                title = None
            get_generations_plot(data, ax=ax, num_scalar_funcs=scalar_funcs, ylabel=ylabel, title=title)
            ax.get_legend().remove()
            if i == len(df['TotalGenerations'].unique()) - 1:
                ax.set_xlabel('Profit')
    plt.tight_layout()
    plt.show()
