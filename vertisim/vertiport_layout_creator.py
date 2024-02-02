import pandas as pd
import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
from .utils.layout_layer_for_unfolded import save_vertiport_layout_layer_for_unfolded
from .utils.read_files import read_input_file
from .utils.helpers import check_if_dataframe_is_empty, convert_to_list
from .utils.distance import haversine_dist
from collections import defaultdict


class VertiportLayout:
    def __init__(self, file_path: str, output_folder_path: str, layout_type: str, vertiport_id: str):
        self.file_path = file_path
        self.output_folder_path = output_folder_path
        self.layout_type = layout_type
        self.vertiport_id = vertiport_id
        self.G = nx.Graph()
        self.node_pos = {}
        self.edge_list = []
        self.node_sizes = []
        self.color_map = []

        self.read_vertiport_layout()
        self.add_fato_nodes_to_graph()
        self.add_parking_pad_nodes_to_graph()
        self.add_taxiway_nodes_to_graph()
        self.add_nodes_to_graph()
        self.edge_ids = self.add_edges_to_graph()
        self.node_distances = self.calculate_distances_matrix()
        self.num_fato = self.get_num_fato()
        self.num_parking_pad = self.get_num_parking_pad()
        self.num_taxiway_nodes = self.get_num_taxi_nodes()
        self.num_edges = self.get_num_edges()
        self.terminal_airspace_radius = self.compute_terminal_airspace_radius()
        self.vertiport_element_locations = self.get_structural_entity_locations()
        self.structural_entity_groups = self.get_structural_entity_groups()
        # save_vertiport_layout_layer_for_unfolded(df1=self.fato_lat_lngs,
        #                                          df2=self.park_lat_lngs,
        #                                          df3=self.inters_lat_lngs,
        #                                          height=self.vertiport_height,
        #                                          layout_type=self.layout_type,
        #                                          output_folder_path=self.output_folder_path)

    def import_vertiport_layout(self) -> pd.DataFrame:
        """
        Imports the vertiport layout
        """
        vertiport_layout_file = read_input_file(file_path=self.file_path, sheet_name=self.layout_type)
        if check_if_dataframe_is_empty(vertiport_layout_file):
            raise IOError('Vertiport layout file is empty')
        return vertiport_layout_file

    def read_vertiport_layout(self) -> None:
        """
        Get values from input file
        """
        df = self.import_vertiport_layout()
        self.fato_ids = list(df[df['fato_nodes'].notna()]['fato_nodes'])
        self.fato_nodes = df[df['fato_nodes'].notna()][['fato_nodes', 'fato_x', 'fato_y', 'fato_diameter_ft']]
        self.approach_fix_lat_lng = df[df['fato_nodes'].notna()][['fato_nodes', 'approach_fix_nodes', 'approach_fix_lat', 
                                                                  'approach_fix_lon']]
        # !!! This function only supports one approach fix node. This is used in vertiport manager.
        self.approach_fix_node = self.approach_fix_lat_lng.approach_fix_nodes.iloc[0]
        self.departure_fix_lat_lng = df[df['fato_nodes'].notna()][
            ['fato_nodes', 'departure_fix_nodes', 'departure_fix_lat', 'departure_fix_lon']]
        # !!! This function only supports one approach fix node. This is used in vertiport manager.
        self.departure_fix_node = self.departure_fix_lat_lng.departure_fix_nodes.iloc[0]

        self.parking_pad_ids = list(df[df['parking_pad_nodes'].notna()]['parking_pad_nodes'])
        self.parking_pad_nodes = df[df['parking_pad_nodes'].notna()][
            ['parking_pad_nodes', 'parking_pad_x', 'parking_pad_y', 'parking_pad_diameter_ft']]
        self.taxiway_node_ids = list(df[df['taxi_ramp_intersection_nodes'].notna()]['taxi_ramp_intersection_nodes'])
        self.taxiway_nodes = df[df['taxi_ramp_intersection_nodes'].notna()][
            ['taxi_ramp_intersection_nodes', 'inters_x', 'inters_y', 'inters_diameter_ft']]
        self.edges = df[df['origin_node'].notna()][['origin_node', 'destination_node']]

        self.fato_lat_lngs = df[df['fato_nodes'].notna()][['fato_nodes', 'fato_lat', 'fato_lon', 'fato_diameter_ft']]
        self.fato_lat_lngs.set_index('fato_nodes', inplace=True)

        self.park_lat_lngs = df[df['parking_pad_nodes'].notna()][[
            'parking_pad_nodes', 'parking_pad_lat', 'parking_pad_lon', 'parking_pad_diameter_ft']]
        self.park_lat_lngs.set_index('parking_pad_nodes', inplace=True)

        self.inters_lat_lngs = df[df['taxi_ramp_intersection_nodes'].notna()][[
            'taxi_ramp_intersection_nodes', 'inters_lat', 'inters_lon', 'inters_diameter_ft']]
        self.inters_lat_lngs.set_index('taxi_ramp_intersection_nodes', inplace=True)

        self.vert_door_lat_lngs = df[df['vert_door_nodes'].notna()][['vert_door_nodes', 'vert_door_lat', 'vert_door_lon']]
        self.vert_door_node = self.vert_door_lat_lngs.vert_door_nodes.iloc[0]        
        self.vert_door_lat_lngs.set_index('vert_door_nodes', inplace=True)

        self.waiting_room_lat_lngs = df[df['waiting_room_nodes'].notna()][
            ['waiting_room_nodes', 'waiting_room_lat', 'waiting_room_lon']]
        self.waiting_room_node = self.waiting_room_lat_lngs.waiting_room_nodes.iloc[0]          
        self.waiting_room_lat_lngs.set_index('waiting_room_nodes', inplace=True)      

        self.boarding_gate_lat_lngs = df[df['boarding_gate_nodes'].notna()][
            ['boarding_gate_nodes', 'boarding_gate_lat', 'boarding_gate_lon']]
        self.boarding_gate_node = self.boarding_gate_lat_lngs.boarding_gate_nodes.iloc[0]           
        self.boarding_gate_lat_lngs.set_index('boarding_gate_nodes', inplace=True)

        self.vertiport_height = df['height_m'].iloc[0]

    def get_structural_entity_locations(self) -> dict:
        """
        Returns a dictionary of vertiport elements and their locations
        """
        vertiport_elements = defaultdict(dict)
        for node in self.fato_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)

        for node in self.park_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)

        for node in self.inters_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)

        for node in self.vert_door_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)

        for node in self.boarding_gate_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)

        for node in self.waiting_room_lat_lngs.reset_index().values:
            vertiport_elements[node[0]] = (node[1], node[2], self.vertiport_height)
        return vertiport_elements
    
    def get_structural_entity_groups(self) -> dict:
        structural_entity_groups = defaultdict(list)
        structural_entity_groups['fato'] = convert_to_list(self.fato_ids)
        structural_entity_groups['parking_pad'] = convert_to_list(self.parking_pad_ids)
        structural_entity_groups['taxiway_node'] = convert_to_list(self.taxiway_node_ids)
        structural_entity_groups['vertiport_entrance'] = convert_to_list(self.vert_door_node)
        structural_entity_groups['waiting_room'] = convert_to_list(self.waiting_room_node)
        structural_entity_groups['boarding_gate'] = convert_to_list(self.boarding_gate_node)
        return structural_entity_groups

    def get_num_fato(self) -> int:
        return len(self.fato_ids)

    def get_num_parking_pad(self) -> int:
        return len(self.parking_pad_ids)

    def get_num_taxi_nodes(self) -> int:
        return len(self.taxiway_nodes)

    def get_num_edges(self) -> int:
        return len(self.edges)

    def add_fato_nodes_to_graph(self):
        # Add all of the nodes to the Graph G.
        for node in self.fato_nodes.values:
            self.node_pos[node[0]] = (node[1], node[2])
            self.node_sizes.append(node[3] * 200 * 108 / 60)
            self.color_map.append('C8')

    def add_parking_pad_nodes_to_graph(self):
        for node in self.parking_pad_nodes.values:
            self.node_pos[node[0]] = (node[1], node[2])
            self.node_sizes.append(node[3] * 200)
            self.color_map.append('C2')

    def add_taxiway_nodes_to_graph(self):
        for node in self.taxiway_nodes.values:
            self.node_pos[node[0]] = (node[1], node[2])
            self.node_sizes.append(10)
            self.color_map.append('C1')

    def add_nodes_to_graph(self):
        self.G.add_nodes_from(self.node_pos)

    def add_edges_to_graph(self):
        # Add edges to the Graph G.
        for node in self.edges.values:
            self.edge_list.append((node[0], node[1]))
            distance = np.sqrt((self.node_pos[node[0]][0] - self.node_pos[node[1]][0]) ** 2 +
                               (self.node_pos[node[0]][1] - self.node_pos[node[1]][1]) ** 2)
            self.G.add_edge(node[0], node[1], weight=distance)
        return list(self.G.edges)
    
    def compute_terminal_airspace_radius(self):
        # Compute the terminal airspace radius
        return haversine_dist(self.fato_lat_lngs['fato_lat'].iloc[0], 
                              self.fato_lat_lngs['fato_lon'].iloc[0],
                              self.approach_fix_lat_lng['approach_fix_lat'].iloc[0], 
                              self.approach_fix_lat_lng['approach_fix_lon'].iloc[0],
                              unit='mile')

    def calculate_distances_matrix(self):
        # Calculate the distances matrix
        node_distances = pd.DataFrame(index=self.G.nodes, columns=self.G.nodes)
        for source in self.G.nodes:
            for target in self.G.nodes:
                try:
                    node_distances.loc[source, target] = round(nx.shortest_path_length(
                        self.G, source=source, target=target, weight='weight'))
                except:
                    continue
        return node_distances

    # def plot_vertiport_layout(self):
    #     fig, ax = plt.subplots(figsize=(20, 6))
    #     nx.draw_networkx(self.G, pos=self.node_pos, node_size=self.node_sizes, node_color=self.color_map, ax=ax)
    #     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    #     plt.grid('on')
    #     plt.axis('on')
    #     plt.show()
