import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import json
from matplotlib.collections import PatchCollection

def plot_inter_intra_network_connectivity(intra_network_connectivity, inter_network_connectivity, ordered_networks, network_json_path, network_names):
    """
    Plots Inter/Intra-Network Connectivity

    Inputs:
    - Average correlation matrix of intra-network connectivity
    - Average correlation matrix of inter-network connectivity
    - Key of network numbers mapped to network names
    - Path to the network names JSON file (Yeo 7 networkds commonly used)
    - network names

    Parameters:
    - Subplot with two axes (ax1 and ax2) is created for intra-network and inter-network connectivity
    - Horizontal bar plot (barh) is created for intra-network connectivity in ax1
    - Heatmap is created for inter-network connectivity in ax2

    Returns:
    - Combined plot of Intra and Inter-network connectivity with the networks and network names
    """
    matrix_df = pd.DataFrame(np.zeros((len(ordered_networks), len(ordered_networks))), index=ordered_networks, columns=ordered_networks)
    for key, value in inter_network_connectivity.items():
        network_i, network_j = key
        matrix_df.loc[network_i, network_j] = value
        matrix_df.loc[network_j, network_i] = value
    if 0 in matrix_df.index:
        matrix_df.drop(0, axis=0, inplace=True)
        matrix_df.drop(0, axis=1, inplace=True)
    mask = np.triu(np.ones_like(matrix_df, dtype=bool))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [0.4, 1]})

    network_labels = [network_names.get(network, f'Network {network}') for network in matrix_df.index]
    intra_network_values = [intra_network_connectivity[network] for network in matrix_df.index]

    ax1.barh(network_labels, intra_network_values, color='blue')
    ax1.set_title('Intra-Network Connectivity')
    ax1.set_xlim(0, max(intra_network_connectivity.values()) + 0.1)
    ax1.set_xlabel('Average Correlation')
    ax1.grid(axis='x')
    sns.heatmap(matrix_df, cmap="coolwarm", annot=True, linewidths=.5, mask=mask, ax=ax2, cbar_kws={"shrink": 0.8})
    ax2.set_xticks(np.arange(0.5, len(matrix_df.index)))
    ax2.set_xticklabels(network_labels, rotation=45)
    ax2.set_yticks(np.arange(0.5, len(matrix_df.index)))
    ax2.set_yticklabels(network_labels, rotation=0)
    ax2.set_title("Inter-Network Connectivity")
    plt.tight_layout()
    plt.show()
    return matrix_df, network_labels


def plot_correlation_matrix(matrix_df,network_labels, max_size=2500):
    # Calculate the size and color of each dot based on the correlation value
    x, y = np.meshgrid(np.arange(matrix_df.shape[0]), np.arange(matrix_df.shape[1]))
    x = x.flatten()
    y = y.flatten()
    size = matrix_df.flatten() * max_size
    color = matrix_df.flatten()

    # Makes a mask to show only bottom triangle of data
    mask = np.tril(np.ones_like(matrix_df, dtype=bool))
    x_masked = x[mask.flatten()]
    y_masked = y[mask.flatten()]
    size_masked = size[mask.flatten()]
    color_masked = color[mask.flatten()]
    plt.scatter(x_masked+0.4, y_masked+0.4, s=size_masked, c=color_masked, cmap='coolwarm',vmin=0.1)

    # Add labels and title
    plt.title("Inter-Network Connectivity")
    #plt.xticks(np.arange(len(network_labels)), labels=network_labels, rotation=45)
    #plt.yticks(np.arange(len(network_labels)), labels=network_labels)
    plt.ylim(len(network_labels), -1)

    x_ticks = np.unique(x_masked)
    y_ticks = np.unique(y_masked)

    # Add grid lines at the midpoints between circle positions
    plt.xticks(np.arange(len(network_labels)) +0.4, labels=network_labels, rotation=90, ha='center', va='top')
    plt.yticks(np.arange(len(network_labels)) +0.4, labels=network_labels, va='center')

    #plt.grid(True)
    plt.colorbar()
    plt.show()

    #Add in chord plot (that thresholds)