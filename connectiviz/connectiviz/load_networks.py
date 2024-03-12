import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import json

def create_mapped_timeseries(original_timeseries, network_name, network_dictionary, subregions):
     """
     Parameters:
     - Reads a JSON file (network_dictionary) containing network names (Yeo_7network_names)
     - Reads a CSV file (subregions) containing network labels and their mappings (network_mapping)
     - Maps the values in the first column of the original_timeseries based on the network labels and stores the result in mapped_timeseries
     - Creates new_timeseries by selecting columns from the 5th column onward from the original_timeseries
     - Adds the mapped timeseries as a new column named with the provided network_name
     - Extracts unique values from the 'Yeo_7network' column of the new_timeseries and stores them in yeo_networks
     Returns:
     - new_timeseries
     - yeo_networks
     - loaded Yeo_7network_names
     """
     # Import network dictionary
     with open(network_dictionary) as file:
         Yeo_7network_names = json.load(file)
     networks = pd.read_csv(subregions)
     network_mapping = networks.set_index('Label')[f'{network_name}'].to_dict()
     mapped_timeseries = original_timeseries.iloc[:, 0].map(network_mapping)
     new_timeseries = original_timeseries.iloc[:, 4:]
     new_timeseries[f'{network_name}'] = mapped_timeseries
     yeo_networks = new_timeseries['Yeo_7network'].unique()
     return new_timeseries, yeo_networks, Yeo_7network_names


def load_network_names(json_filepath):
    """
    Loads the network names from a JSON file and converts keys from strings to integers
    """
    with open(json_filepath, 'r') as file:
        data = json.load(file)
        return {int(k): v for k, v in data.items()}
