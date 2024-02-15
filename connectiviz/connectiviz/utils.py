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
import holoviews as hv
from holoviews import opts
from holoviews import dim
from bokeh.sampledata.les_mis import data

hv.extension('bokeh')
hv.output(size=200)

def filter_by_region_threshold(p_value_df, effect_sizes_df, selected_region, threshold):
    if 'region' not in p_value_df.columns or 'region' not in effect_sizes_df.columns:
        raise ValueError("Both dataframes must have a 'region' column.")

    # Filter both dataframes for the selected region
    p_region_row = p_value_df[p_value_df['region'] == selected_region]
    effect_region_row = effect_sizes_df[effect_sizes_df['region'] == selected_region]

    if p_region_row.empty or effect_region_row.empty:
        print(f"No data found for region: {selected_region}")
        return pd.DataFrame()

    # Find columns where the p-value is below the threshold in p_value_df
    cols_to_check = p_value_df.columns.difference(['region', 'Yeo_17network'])
    cols_below_threshold = ['region', 'Yeo_17network']  # Start with 'region' and 'Yeo_17network' in the list to keep them in the output

    for col in cols_to_check:
        if p_region_row[col].values[0] < threshold:  # Note the use of < rather than >
            cols_below_threshold.append(col)

    # Select the same columns from the effect_sizes_df for the selected region
    filtered_effect_sizes = effect_region_row[cols_below_threshold]

    return filtered_effect_sizes


#------------------------------------------------------------------------------------------
def filter_p_values_by_region_threshold(p_value_df, selected_region, threshold):
    if 'region' not in p_value_df.columns:
        raise ValueError("The dataframe does not have a 'region' column.")
    
    # Filter the dataframe for the selected region
    p_region_row = p_value_df[p_value_df['region'] == selected_region]

    if p_region_row.empty:
        print(f"No data found for region: {selected_region}")
        return pd.DataFrame()

    # Select only numeric columns (excluding 'region' and 'Yeo_17network' if they are not numeric)
    numeric_cols = p_value_df.select_dtypes(include=[np.number]).columns.difference(['region', 'Yeo_17network'])

    # Find columns where the p-value is below the threshold
    cols_below_threshold = numeric_cols[(p_region_row[numeric_cols] < threshold).iloc[0]]

    # Include the 'region' and 'Yeo_17network' columns regardless of their p-values
    cols_to_keep = ['region', 'Yeo_17network'] + cols_below_threshold.tolist()

    # Return the filtered dataframe with the columns below the threshold
    filtered_p_values = p_value_df.loc[p_value_df['region'] == selected_region, cols_to_keep]

    return filtered_p_values


#------------------------------------------------------------------------------------------


def generate_edges(df, network_name):
    # Get the region name from the 'region' column
    region_name = df['region'].values[0]

    # Create a list of all other column names, excluding 'region' and 'Yeo_17networks'
    other_columns = [col for col in df.columns if col not in ['region', network_name]]

    # Create the edges by pairing the region name with each of the other column names
    edges = [(region_name, other_col) for other_col in other_columns]

    return edges

#------------------------------------------------------------------------------------------

def generate_network_mapped_edges(original_df, filtered_df, network_name):
    region_to_network = pd.Series(original_df[network_name].values, index=original_df.region).to_dict()    
    edges = [(region_to_network.get(region), region_to_network.get(other_region)) for region, other_region in generate_edges(filtered_df, network_name)]

    return edges

#------------------------------------------------------------------------------------------

def generate_edge_weights(df, network_name):
    edge_weights = df.drop(['region', network_name], axis=1).values.flatten()
    return edge_weights

#------------------------------------------------------------------------------------------

def fc_chord_plot(network_dictionary, edges, edge_weights, selected_region, selected_threshold):
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

#------------------------------------------------------------------------------------------

# Function to calculate Cohen’s d
def cohens_d(group1, group2):
    # Calculate the size of samples
    n1, n2 = len(group1), len(group2)
    # Calculate the variance of the samples
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    # Calculate Cohen’s d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d
# Assuming dataframes_rATL and dataframes_ctrl are lists of dataframes
# And each dataframe has the same shape
# Get the shape of the dataframes (assuming all dataframes have the same shape)
num_rows, num_cols = dataframes_rATL[0].shape
# Initialize matrices to hold the p-values, t-statistics, and effect sizes
p_values_matrix = np.zeros((num_rows, num_cols))
t_stats_matrix = np.zeros((num_rows, num_cols))
effect_sizes_matrix = np.zeros((num_rows, num_cols))
# Get the indices for the upper triangle of the matrix, excluding the diagonal
rows, cols = np.triu_indices(num_rows, k=1)
# Perform t-tests and fill in the matrices
for (i, j) in zip(rows, cols):
    # Collect all values from position (i, j) across all dataframes for each group
    group1_values = [df.iloc[i, j] for df in dataframes_ctrl]
    group2_values = [df.iloc[i, j] for df in dataframes_rATL]
    # Perform the t-test between the two groups of values
    t_stat, p_val = stats.ttest_ind(group1_values, group2_values)
    # Calculate Cohen’s d
    d = cohens_d(group1_values, group2_values)
    # Fill in the p-values, t-statistics, and effect sizes matrices
    p_values_matrix[i, j] = p_val
    p_values_matrix[j, i] = p_val  # mirror the values since the matrix is symmetric
    t_stats_matrix[i, j] = t_stat
    t_stats_matrix[j, i] = t_stat
    effect_sizes_matrix[i, j] = d
    effect_sizes_matrix[j, i] = d