import seaborn as sns
import matplotlib.pyplot as plt


def show_igd_heatmap(df):
    heatmap_igd_df = df.query('Generation == TotalGenerations')
    heatmap_igd_df = heatmap_igd_df[['Scalar_funcs', 'TotalGenerations', 'IGD']]
    heatmap_igd_df = heatmap_igd_df.groupby(['Scalar_funcs', 'TotalGenerations']).first()
    heatmap_igd_df = heatmap_igd_df.unstack()
    heatmap_igd_df = heatmap_igd_df.droplevel(0, axis=1)
    ax = sns.heatmap(heatmap_igd_df, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'IGD'}, linewidths=0.5)
    ax.set_ylabel('Scalarizing Functions')
    ax.set_title('IGD for different number of Scalarizing Functions and Generations')
    plt.tight_layout()
