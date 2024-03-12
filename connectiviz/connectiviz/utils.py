# import useful library and tools
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from scipy.stats import ttest_ind
import scipy.stats as stats
from mne_connectivity.viz import plot_connectivity_circle
import json
import argparse
import pandas as pd


def cohens_d(group1, group2):
    ''' Calculate Cohen’s d for independent samples'''
    # Calculate the size of samples
    n1, n2 = len(group1), len(group2)
    # Calculate the variance of the samples
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    # Calculate Cohen’s d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def rename_columns_by_region(df_to_rename, df_with_regions):
    ''' Rename the columns of df_to_rename with the regions in df_with_regions.'''

    if 'region' not in df_with_regions.columns:
        raise ValueError("df_with_regions does not have a 'region' column.")

    if len(df_with_regions['region']) != len(df_to_rename.columns):
        raise ValueError("The number of regions does not match the number of columns to rename.")

    df_to_rename.columns = df_with_regions['region'].values

    return df_to_rename


def OLD_fc_chord_plot(network_dictionary, edges, edge_weights, selected_region, selected_threshold):
    # Initialize the connectivity matrix
    num_networks = len(network_dictionary)
    connectivity_matrix = np.zeros((num_networks, num_networks))

    # Fill in the weights in the connectivity matrix
    for (i, j), weight in zip(edges, edge_weights):
        i_adj = i - 1  # Adjust for 0-indexing
        j_adj = j - 1  # Adjust for 0-indexing
        connectivity_matrix[i_adj, j_adj] = weight
        connectivity_matrix[j_adj, i_adj] = weight  # if undirected/bidirectional

    # Sort your network names based on the natural integer sort order of the dictionary keys
    sorted_network_names = [network_dictionary[key] for key in sorted(network_dictionary)]

    # Generate colors for each network
    label_colors = plt.cm.hsv(np.linspace(0, 1, num_networks))

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85)

    # Plot the connectivity circle
    plot_connectivity_circle(connectivity_matrix, sorted_network_names,
                             node_colors=label_colors, node_edgecolor='white',
                             fontsize_names=10, textcolor='white',
                             node_linewidth=2, colormap='hot', vmin=0, vmax=np.max(edge_weights),
                             linewidth=1.5, colorbar=True,
                             title=f'Functional Connectivity for {selected_region} at {selected_threshold} threshold',
                             fig=fig, subplot=(1, 1, 1), show=False)

    # Adjust layout to make room for the colorbar
    fig.canvas.draw()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #Save the plot to output folder
    #plt.savefig(f'{output_dir}/test.png', dpi=300)

    # Show the plot
    plt.show()
