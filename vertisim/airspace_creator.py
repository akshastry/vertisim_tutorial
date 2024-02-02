from .utils.distance import haversine_dist
import numpy as np
from typing import List, Dict
from .utils.read_files import read_input_file
from collections import defaultdict
import pandas as pd

class AirspaceCreator:
    def __init__(self, vertiports: Dict, airspace_params: Dict, vertiport_distances: pd.DataFrame = None, cruise_altitude: float = 450):
        self.vertiports = vertiports
        self.airspace_params = airspace_params
        self.vertiport_distances = vertiport_distances
        self.cruise_altitude = cruise_altitude
        if airspace_params['airspace_layout_file_path'] is None:
            self.num_waypoints_per_airlink = self.determine_num_waypoints()
            self.airlinks_start_end_points = self.get_airlinks_start_end_points()
            self.waypoint_ids, self.flight_directions = self.create_waypoint_ids()
            self.waypoints = self.create_waypoints()
            self.waypoint_locations = self.match_waypoint_names_locations()
        else:
            self.waypoint_ids, self.waypoint_locations, self.flight_directions = self.read_waypoint_locations_from_file()
            print("Success: Airspace layout file imported.")


    def read_waypoint_locations_from_file(self):
        waypoints = read_input_file(file_path=self.airspace_params['airspace_layout_file_path'])
        waypoint_ids = defaultdict(list)
        waypoint_locations = defaultdict(dict)
        flight_directions = set()
        for _, row in waypoints.iterrows():
            waypoint_ids[row['flight_direction']].append(row['waypoint_id'])       
            waypoint_locations[row['waypoint_id']] = [row['latitude'], row['longitude'], row['altitude']]
            flight_directions.add(row['flight_direction'])
        return waypoint_ids, waypoint_locations, flight_directions

    
    @staticmethod
    def generate_waypoints(start_end_points, num_waypoints, cruise_altitude, origin_destination_vertiport_locs, origin_altitude, destination_altitude):
        lat1, lon1, lat2, lon2 = start_end_points # Approach departures fixes
        o_vert_lat, o_vert_lon, d_vert_lat, d_vert_lon = origin_destination_vertiport_locs

        # Calculate haversine distance between the two points
        distance_miles = haversine_dist(lat1, lon1, lat2, lon2, unit='mile')

        # Convert distance to radians
        c = distance_miles / 3958

        # Convert latitude and longitude to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        waypoints = []
        for i in range(1, num_waypoints + 1 - 4):
            f = i / (num_waypoints + 1)
            A = np.sin((1 - f) * c) / np.sin(c)
            B = np.sin(f * c) / np.sin(c)
            x = A * np.cos(lat1_rad) * np.cos(lon1_rad) + B * np.cos(lat2_rad) * np.cos(lon2_rad)
            y = A * np.cos(lat1_rad) * np.sin(lon1_rad) + B * np.cos(lat2_rad) * np.sin(lon2_rad)
            z = A * np.sin(lat1_rad) + B * np.sin(lat2_rad)
            lat_rad = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
            lon_rad = np.arctan2(y, x)
            lat, lon = np.degrees(lat_rad), np.degrees(lon_rad)
            waypoints.append((lat, lon, cruise_altitude))
        return [(o_vert_lat, o_vert_lon, origin_altitude+15)]+ [(lat1, lon1, origin_altitude+60)] + waypoints + [(lat2, lon2, destination_altitude+60)] + [(d_vert_lat, d_vert_lon, destination_altitude+15)] # HACK: 60 is the altitude of the approach departure fix. This is according to one model. Needs to change.

    
    def determine_num_waypoints(self) -> Dict: # vertiport_distances: pd.DataFrame
        """
        Computes the required number of waypoints for each airlink in the vertiport_distances DataFrame.
        """
        num_waypoints_per_airlink = defaultdict(dict)
        for _, row in self.vertiport_distances.iterrows():
            origin_terminal_airspace_radius = self.vertiports[row['origin_vertiport_id']].vertiport_layout.terminal_airspace_radius
            destination_terminal_airspace_radius = self.vertiports[row['destination_vertiport_id']].vertiport_layout.terminal_airspace_radius
            airlink_length = row['distance_miles'] - origin_terminal_airspace_radius - destination_terminal_airspace_radius
            num_waypoints = int(airlink_length / self.airspace_params['airlink_segment_length_mile']) - 1
            num_waypoints_per_airlink[row['origin_vertiport_id'] + '_' + row['destination_vertiport_id']] = num_waypoints # + 4 # Add 2 for the origin and destination vertiports and 2 for the departure and arrival fix
        return num_waypoints_per_airlink

    def get_airlinks_start_end_points(self):
        airlinks_start_end_endpoints = defaultdict(dict)
        for _, row in self.vertiport_distances.iterrows():
            origin_departure_fix_lat = self.vertiports[row['origin_vertiport_id']].vertiport_layout.departure_fix_lat_lng['departure_fix_lat'].iloc[0]
            origin_departure_fix_lng = self.vertiports[row['origin_vertiport_id']].vertiport_layout.departure_fix_lat_lng['departure_fix_lon'].iloc[0]
            dest_arrival_fix_lat = self.vertiports[row['destination_vertiport_id']].vertiport_layout.approach_fix_lat_lng['approach_fix_lat'].iloc[0]
            dest_arrival_fix_lng = self.vertiports[row['destination_vertiport_id']].vertiport_layout.approach_fix_lat_lng['approach_fix_lon'].iloc[0]
            airlinks_start_end_endpoints[row['origin_vertiport_id'] + '_' + row['destination_vertiport_id']] = (origin_departure_fix_lat, 
                                                                                                                origin_departure_fix_lng, 
                                                                                                                dest_arrival_fix_lat, 
                                                                                                                dest_arrival_fix_lng)
        return airlinks_start_end_endpoints
    
    
    def get_origin_destination_vertiport_locs(self):
        origin_destination_vertiport_locs = defaultdict(dict)
        for _, row in self.vertiport_distances.iterrows():
            origin_vertiport_lat = self.vertiports[row['origin_vertiport_id']].vertiport_layout.fato_lat_lngs['fato_lat'].iloc[0]
            origin_vertiport_lng = self.vertiports[row['origin_vertiport_id']].vertiport_layout.fato_lat_lngs['fato_lon'].iloc[0]
            dest_vertiport_lat = self.vertiports[row['destination_vertiport_id']].vertiport_layout.fato_lat_lngs['fato_lat'].iloc[0]
            dest_vertiport_lng = self.vertiports[row['destination_vertiport_id']].vertiport_layout.fato_lat_lngs['fato_lon'].iloc[0]
            origin_destination_vertiport_locs[row['origin_vertiport_id'] + '_' + row['destination_vertiport_id']] = (origin_vertiport_lat,
                                                                                                                    origin_vertiport_lng,
                                                                                                                    dest_vertiport_lat,
                                                                                                                    dest_vertiport_lng)
        return origin_destination_vertiport_locs

    def create_waypoint_ids(self):
        waypoint_ids = defaultdict(list)
        flight_directions = set()
        for flight_direction, num_waypoints in self.num_waypoints_per_airlink.items():
            flight_directions.add(flight_direction)
            origin = flight_direction.split('_')[0]
            destination = flight_direction.split('_')[1]
            num_waypoints = num_waypoints + 4
            for i in range(num_waypoints):
                if i == num_waypoints-1:
                    waypoint_ids[flight_direction].append(f'{destination}_hover_fix')
                elif i == num_waypoints-2:
                    waypoint_ids[flight_direction].append(f'{destination}_approach_fix')
                elif i == num_waypoints-3:
                    waypoint_ids[flight_direction].append(f'{destination}_holding_unit')
                elif i == 0:
                    waypoint_ids[flight_direction].append(f'{origin}_hover_fix')
                elif i == 1:
                    waypoint_ids[flight_direction].append(f'{origin}_departure_fix')
                else:
                    waypoint_ids[flight_direction].append(f'{flight_direction}_waypoint_{str(i-1)}')
        return waypoint_ids, flight_directions


    def create_waypoints(self):
        waypoints = defaultdict(list)
        origin_destination_vertiport_locs = self.get_origin_destination_vertiport_locs()
        for flight_direction, start_end_points in self.airlinks_start_end_points.items():
            num_waypoints = len(self.waypoint_ids[flight_direction])
            origin = flight_direction.split('_')[0]
            destination = flight_direction.split('_')[1]
            origin_altitude = self.vertiports[origin].vertiport_layout.vertiport_height                    
            destination_altitude = self.vertiports[destination].vertiport_layout.vertiport_height
            waypoints[flight_direction] = AirspaceCreator.generate_waypoints(start_end_points, 
                                                                            num_waypoints,
                                                                            self.cruise_altitude,
                                                                            origin_destination_vertiport_locs[flight_direction],
                                                                            origin_altitude,
                                                                            destination_altitude)
        return waypoints

    def match_waypoint_names_locations(self):
        waypoint_locations = defaultdict(dict)
        for flight_direction, waypoint_id_list in self.waypoint_ids.items():
            for i, waypoint_id in enumerate(waypoint_id_list):
                waypoint_locations[waypoint_id] = self.waypoints[flight_direction][i]
        return waypoint_locations