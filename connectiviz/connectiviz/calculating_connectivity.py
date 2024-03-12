import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import json
from matplotlib.collections import PatchCollection


#Intra-Network Connectivity
def calculate_intra_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
    """
    Intra-Network Connectivity

    Inputs:
    - orginal single-subject timeseries
    - new timeseries (discarding the first four columns of orig_timeseries and adding a new column with the mapped values of network labels)
    - path to the network names
    - network names
    
    Parameters:
    - Key of network numbers mapped to network names
    - For each network, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
    - Calculates the correlation matrix for the selected data using network_data.transpose().corr() then gets the average correlations

    Returns:
    - Intra-network connectivity average correlation matrix
    """
    intra_network_connectivity = {}
    ordered_networks = network_names.keys()
    for network in ordered_networks:
        if network != 0:
            network_data = original_timeseries[new_timeseries['Yeo_7network'] == network]
            correlation_matrix = network_data.transpose().corr()
            average_corr = np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)])
            intra_network_connectivity[network] = average_corr
    return intra_network_connectivity


#Inter-Network Connectivity
def calculate_inter_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
    """
    Inter-Network Connectivity
    
    Inputs:
    - orginal single-subject timeseries
    - new timeseries
    - path to the network names
    - network names

    Parameters:
    - Key of network numbers mapped to network names (ordered_networks)
    - Iterates over pairs of networks (network_i, network_j)
    - For each network pair, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
    - Calculates the correlation between each pair of rows (one from each network) and then gets the average correlations

    Returns:
    - Inter-network connectivity average correlation matrix
    """
    inter_network_connectivity = {}
    ordered_networks = network_names.keys()
    for i, network_i in enumerate(ordered_networks):
        for j, network_j in enumerate(ordered_networks):
            if network_i != network_j and network_i != 0 and network_j != 0:
                data_i = original_timeseries[new_timeseries['Yeo_7network'] == network_i]
                data_j = original_timeseries[new_timeseries['Yeo_7network'] == network_j]
                correlation_values = []
                for index_i, row_i in data_i.iterrows():
                    for index_j, row_j in data_j.iterrows():
                        correlation_values.append(row_i.corr(row_j))
                average_corr = np.mean(correlation_values)
                key = (network_i, network_j)
                inter_network_connectivity[key] = average_corr
    return inter_network_connectivity

#Inter-Network Connectivity Counter 
def count_inter_network_connectivity(original_timeseries, new_timeseries, network_json_path, network_names):
    """
    Inter-Network Connectivity
    
    Inputs:
    - orginal single-subject timeseries
    - new timeseries
    - path to the network names
    - network names

    Parameters:
    - Key of network numbers mapped to network names (ordered_networks)
    - Iterates over pairs of networks (network_i, network_j)
    - For each network pair, it selects the relevant data from original_timeseries based on the network labels in new_timeseries['Yeo_7network']
    - Calculates the correlation between each pair of rows (one from each network) and then gets the average correlations

    Returns:
    - Inter-network connectivity average correlation matrix
    """
    inter_network_connectivity = {}
    inter_network_counter = {}  # Counter for the number of correlation values
    ordered_networks = network_names.keys()
    for i, network_i in enumerate(ordered_networks):
        for j, network_j in enumerate(ordered_networks):
            if network_i != network_j and network_i != 0 and network_j != 0:
                data_i = original_timeseries[new_timeseries['Yeo_7network'] == network_i]
                data_j = original_timeseries[new_timeseries['Yeo_7network'] == network_j]
                correlation_values = []

                for index_i, row_i in data_i.iterrows():
                    for index_j, row_j in data_j.iterrows():
                        correlation_values.append(row_i.corr(row_j))

                for index_i, row_i in data_i.iterrows():
                    for index_j, row_j in data_j.iterrows():
                        correlation_values.append(row_i.corr(row_j))
                average_corr = np.mean(correlation_values)
                key = (network_i, network_j)
                inter_network_connectivity[key] = average_corr
                # Update the counter
                if key in inter_network_counter:
                    inter_network_counter[key] += len(correlation_values)
                else:
                    inter_network_counter[key] = len(correlation_values)
    return inter_network_counter

connectivity_count = count_inter_network_connectivity()