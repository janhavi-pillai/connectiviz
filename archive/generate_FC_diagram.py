# import useful library and tools
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from mne_connectivity.viz import plot_connectivity_circle
import json
import argparse

#Utils

def rename_columns_by_region(df_to_rename, df_with_regions):

    if 'region' not in df_with_regions.columns:
        raise ValueError("df_with_regions does not have a 'region' column.")

    if len(df_with_regions['region']) != len(df_to_rename.columns):
        raise ValueError("The number of regions does not match the number of columns to rename.")

    df_to_rename.columns = df_with_regions['region'].values

    return df_to_rename

#------------------------------------------------------------------------------------------

def filter_by_region_threshold(df, network_name, selected_region, selected_threshold):
    
    if 'region' not in df.columns:
        raise ValueError("The dataframe does not have a 'region' column.")

    region_row = df[df['region'] == selected_region]

    if region_row.empty:
        print(f"No data found for region: {selected_region}")
        return pd.DataFrame()

    columns_to_check = df.columns.difference(['region', network_name])

    cols_above_threshold = []

    for col in columns_to_check:
        if region_row[col].values[0] > selected_threshold:  # Access the first (and only) item in the series
            cols_above_threshold.append(col)

    filtered_df = df.loc[df['region'] == selected_region, cols_above_threshold]
    filtered_df['region'] = selected_region
    filtered_df[network_name] = df.loc[df['region'] == selected_region, network_name].values[0]

    # Ensure 'region' and 'Yeo_17network' columns are at the beginning of t_stats_df
    cols = ['region', network_name]  # these are the columns you want to move to the front
    # Extend the list with the remaining columns that are not 'region' or 'Yeo_17network'
    cols.extend([col for col in filtered_df.columns if col not in cols])

    # Reindex the DataFrame with the new column order
    filtered_df = filtered_df[cols]
    
    return filtered_df

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

def fc_chord_plot(network_dictionary, edges, edge_weights, selected_region, selected_threshold, output_png_path):
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
    plt.savefig(f'{output_png_path}', dpi=300)

    # Show the plot
    # plt.show()

#------------------------------------------------------------------------------------------

def import_subject_corr_coef(csv, network_name, df_networks):
    df = pd.read_csv(csv)
    df = rename_columns_by_region(df, df_networks)
    df['region'] = df_networks['region']
    df[network_name] = df_networks[network_name]
    cols = ['region', network_name]  # these are the columns you want to move to the front
    
    # Extend the list with the remaining columns that are not 'region' or 'Yeo_17network'
    cols.extend([col for col in df.columns if col not in cols])
    df = df[cols]
    return df


#MAIN
def main(csv, network_name, region, threshold, output_png_path):

    if network_name == 'Yeo_17network':
        with open('./Yeo_17network_names.json', 'r') as file:
            network_dict = json.load(file)
    elif network_name == 'Yeo_7network':
        with open('./Yeo_7network_names.json', 'r') as file:
            network_dict = json.load(file)

    df_networks = pd.read_csv('./subregions_Yeo7networks.csv')

    subject = import_subject_corr_coef(csv, network_name, df_networks)

    filtered_df = filter_by_region_threshold(subject, network_name, region, threshold)

    #Generate edges, network mapped edges, and edge weights
    edges = generate_edges(filtered_df, network_name)
    network_edges = generate_network_mapped_edges(subject, filtered_df, network_name)
    edge_weights = generate_edge_weights(filtered_df, network_name)

    #Plot the chord diagram
    fc_chord_plot(network_dict, network_edges, edge_weights, region, threshold, output_png_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate FC Chord Diagrams')
    parser.add_argument('csv', type=str, help='path to your single subject csv file')
    parser.add_argument('network_name', type=str, help='name of the network you want to plot: (Yeo_17network or Yeo_7network)')
    parser.add_argument('region', type=str, help='name of the region you want to select for')
    parser.add_argument('threshold', type=float, help='threshold for the region you want to plot')
    parser.add_argument('output_png_path', type=str, help='path to the output png path')
    args = parser.parse_args()
    main(args.csv, args.network_name, args.region, args.threshold, args.output_png_path)
