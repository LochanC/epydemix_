import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_timeseries_data(df_quantiles, column, quantile): 
    """
    Extracts the time series data for a specific compartment, demographic group, and quantile.

    Parameters:
    -----------
        - df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        - column (str): The name of the column to extract data for.
        - quantile (float): The quantile to extract data for.

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing the time series data for the specified compartment, demographic group, and quantile.
    """
    return df_quantiles.loc[(df_quantiles["quantile"] == quantile)][["date", column]]

def plot_quantiles(df_quantiles, columns, ax=None,
                   lower_q=0.05, upper_q=0.95, show_median=True, 
                   ci_alpha=0.3, title="", show_legend=True, 
                   palette="Set2"):
    """
    Plots the quantiles for a specific compartment and demographic group over time.

    Parameters:
    -----------
        - df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        - compartment (list or str): The names of the compartment to plot data for.
        - demographic_group (list or str or int): The demographic groups to plot data for.
        - ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
        - lower_q (float, optional): The lower quantile to plot (default is 0.05).
        - upper_q (float, optional): The upper quantile to plot (default is 0.95).
        - show_median (bool, optional): Whether to show the median (default is True).
        - ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        - label (str, optional): The label for the median line (default is "").
        - title (str, optional): The title of the plot (default is "").
        - show_legend (bool, optional): Whether to show legend (default is True).
        - palette (str, optional): The color palette for the plot (default is "Set2")
    """
    
    if not isinstance(columns, list):
        columns = [columns]

    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    colors = sns.color_palette(palette, len(columns))
    t = 0
    for column in columns:
        if show_median:
            df_med = get_timeseries_data(df_quantiles, column, 0.5)
            ax.plot(df_med.date, df_med[column].values, color=colors[t], label=column)

        df_q1 = get_timeseries_data(df_quantiles, column, lower_q)
        df_q2 = get_timeseries_data(df_quantiles, column, upper_q)
        ax.fill_between(df_q1.date, df_q1[column].values, df_q2[column].values, alpha=ci_alpha, color=colors[t], linewidth=0.)
        t += 1

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)

    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)