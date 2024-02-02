import os
import pandas as pd
from typing import List, Dict
from .distance import haversine_dist
from .read_files import read_input_file
from .create_vertiport_network import create_fully_connected_vertiport_network
from .helpers import create_save_path

def get_fato_coordinates(vertiport_ids: List, vertiport_layouts: Dict) -> List:
    """Get list of fato coordinates for given vertiport IDs"""
    return [
        (
            vertiport_layouts[vertiport_id].fato_lat_lngs['fato_lat'].values[0],
            vertiport_layouts[vertiport_id].fato_lat_lngs['fato_lon'].values[0],
        )
        for vertiport_id in vertiport_ids
    ]

def create_vertiport_network_file(fato_coords_list: List, vertiport_ids: List, 
                                  original_path: str, network_simulation: bool) -> pd.DataFrame:
    """Create a fully connected vertiport network file"""
    # Create a save path for the vertiport network file using the vertiport ids
    save_path = create_save_path(original_path=original_path,
                                 list_of_keys=vertiport_ids)

    return create_fully_connected_vertiport_network(fato_coordinates=fato_coords_list,
                                                    vertiport_ids=vertiport_ids,
                                                    save_path=save_path,
                                                    network_simulation=network_simulation)

def calc_vertiport_distances(vertiport_network_file_path: str, 
                             vertiport_ids: List,
                             vertiport_layouts: Dict,
                             network_simulation: bool) -> pd.DataFrame:
    """Calculates the distance between origin vertiport and destination vertiports and creates a dictionary of distances
    for each origin vertiport. key should be vertiport_id and value should be haversine distance. Unit is miles or km."""

    if vertiport_network_file_path:
        # Read vertiport network file
        vertiport_network = read_input_file(vertiport_network_file_path)
        
        # If vertiport network file doesn't have all vertiport ids, recreate it
        if not set(vertiport_ids).issubset(set(vertiport_network['origin_vertiport_id'])):
            fato_coords_list = get_fato_coordinates(vertiport_ids, vertiport_layouts)
            vertiport_network = create_vertiport_network_file(fato_coords_list, vertiport_ids, 
                                                              vertiport_network_file_path, network_simulation)
        # print('Success: Vertiport network imported.')
    else:
        fato_coords_list = get_fato_coordinates(vertiport_ids, vertiport_layouts)
        
        # Get the current working directory
        cwd = os.getcwd()
        # Set input location
        input_location = os.path.join(cwd, 'vertisim/vertisim/input/network')

        vertiport_network = create_vertiport_network_file(fato_coords_list=fato_coords_list, 
                                                          vertiport_ids=vertiport_ids, 
                                                          original_path=input_location, 
                                                          network_simulation=network_simulation)
        
        # print('Success: Vertiport network file created.')
        
    # Compute haversine distance and add as a new column to the dataframe
    vertiport_network['distance_miles'] = vertiport_network.apply(
        lambda row: haversine_dist(row['o_vert_lat'], row['o_vert_lon'], row['d_vert_lat'], row['d_vert_lon']), axis=1)

    return vertiport_network
