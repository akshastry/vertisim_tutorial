from typing import List, Tuple
import pandas as pd

def create_fully_connected_vertiport_network(fato_coordinates: List[Tuple[float, float]], 
                                             vertiport_ids: List[str], 
                                             save_path: str,
                                             network_simulation: bool) -> pd.DataFrame:
    """
    Creates a fully vertiport network with given coordinates.
    :param fato_coordinates: list of vertiport coordinates
    ex: 
    fato_coordinates = [
    (34.052110, -118.252680),
    (33.767700, -118.199550),
    (33.94305010754884, -118.40678502696238)
    ]
    :param vertiport_ids: list of vertiport locations
    ex:
    vertiport_ids = ['LADT', 'LongBeach', 'LAX']
    :param save_path: path to save the network
    :param network: if True, it creates a fully connected network. If False, then it creates a unidirectional network.

    :return: pd.DataFrame of vertiport network

        o_vert_lat	o_vert_lon	d_vert_lat	d_vert_lon	origin_vertiport_id	destination_vertiport_id
    0	34.05211	-118.252680	33.76770	-118.199550	LADT	LongBeach
    1	34.05211	-118.252680	33.94305	-118.406785	LADT	LAX
    2	33.76770	-118.199550	34.05211	-118.252680	LongBeach	LADT
    3	33.76770	-118.199550	33.94305	-118.406785	LongBeach	LAX
    4	33.94305	-118.406785	34.05211	-118.252680	LAX	LADT
    5	33.94305	-118.406785	33.76770	-118.199550	LAX	LongBeach
    """
    vertiport_network = pd.DataFrame(columns=['o_vert_lat', 'o_vert_lon', 'd_vert_lat', 'd_vert_lon', 'origin_vertiport_id', 'destination_vertiport_id'])
    if network_simulation:
        for i in range(len(fato_coordinates)):
            for j in range(len(fato_coordinates)):
                if i != j:
                    # Concat vertiport_network with the new row
                    vertiport_network = pd.concat([vertiport_network, 
                                                pd.DataFrame([[fato_coordinates[i][0], 
                                                                fato_coordinates[i][1], 
                                                                fato_coordinates[j][0], 
                                                                fato_coordinates[j][1], 
                                                                vertiport_ids[i], 
                                                                vertiport_ids[j]]], 
                                                                columns=['o_vert_lat', 
                                                                        'o_vert_lon', 
                                                                        'd_vert_lat', 
                                                                        'd_vert_lon', 
                                                                        'origin_vertiport_id', 
                                                                        'destination_vertiport_id'])], 
                                                                        ignore_index=True)
        vertiport_network.to_csv(save_path, index=False)
    else:
        # Single vertiport scenario. The first vertiport in the list is the main vertiport.
        vertiport_network['o_vert_lat'] = [fato_coordinates[0][0] for _ in range(len(fato_coordinates)-1)]
        vertiport_network['o_vert_lon'] = [fato_coordinates[0][1] for _ in range(len(fato_coordinates)-1)]
        vertiport_network['d_vert_lat'] = [x[0] for x in fato_coordinates[1:]]
        vertiport_network['d_vert_lon'] = [x[1] for x in fato_coordinates[1:]]
        vertiport_network['origin_vertiport_id'] = vertiport_ids[0]
        vertiport_network['destination_vertiport_id'] = vertiport_ids[1:]
        vertiport_network.to_csv(f'{save_path}/{len(fato_coordinates)}_destination_single_vertiport.csv', index=False)
    
    return vertiport_network
