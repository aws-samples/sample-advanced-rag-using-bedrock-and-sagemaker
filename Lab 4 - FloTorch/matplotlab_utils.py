import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bar(df, group_by_index, columns_to_plot, show_values=False, title='Grouped Bar Chart', xlabel='Group', ylabel='Value'):
    """
    Generates a grouped bar chart from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_by_index (str): The name of the column to use for grouping on the x-axis.
        columns_to_plot (list): A list of column names to plot as separate groups of bars.
        show_values (bool, optional): Whether to display the values on top of the bars. Defaults to False.
        title (str, optional): The title of the plot. Defaults to 'Grouped Bar Chart'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Group'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Value'.
    """
    unique_groups = df[group_by_index].unique()
    num_groups = len(unique_groups)
    num_cols = len(columns_to_plot)
    bar_width = 0.8 / num_cols  # Adjust width based on the number of columns

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_groups)  # Positions for the groups

    for i, col in enumerate(columns_to_plot):
        values = df.groupby(group_by_index)[col].mean().values
        positions = x + (i * bar_width) - (0.4 - (bar_width / 2)) # Center the groups
        rects = ax.bar(positions, values, bar_width, label=col)

        if show_values:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_groups)
    ax.legend(title='Columns')
    fig.tight_layout()
    plt.show()