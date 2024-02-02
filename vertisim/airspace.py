from typing import List, Dict
import simpy
from .utils.read_files import read_input_file
from .utils.distance import haversine_dist
from collections import defaultdict
import pandas as pd
import numpy as np

class Airspace:
    def __init__(self, env: simpy.Environment(), 
                 holding_unit_capacity: int = 10,
                 airlink_capacity: int = 1,
                 airlink_segment_length_mile_miles: float = 0.25,
                 waypoint_ids: Dict = None,
                 waypoint_locations: Dict = None,
                 flight_directions: List = None) -> None:
        self.env = env
        self.holding_unit_capacity = holding_unit_capacity
        self.airlink_capacity = airlink_capacity
        self.airlink_segment_length_mile_miles = airlink_segment_length_mile_miles
        self.waypoint_ids = waypoint_ids
        self.flight_directions = list(waypoint_ids.keys())
        self.waypoint_locations = waypoint_locations
        self.flight_directions = flight_directions
        self.waypoint_distances = Airspace.compute_distances_bw_waypoints(waypoint_ids=self.waypoint_ids, 
                                                                          waypoint_locations=self.waypoint_locations)
        self.airlink_resources = self.create_airlink_resources()

    def create_airlink_resources(self):
        airlink_resources = defaultdict(dict)
        for flight_direction, waypoint_id_list in self.waypoint_ids.items():
            # origin = flight_direction.split('_')[0]
            # destination = flight_direction.split('_')[1]
            for waypoint_id in waypoint_id_list:
                # Hover fix, Departure fix, Hover fix, FAF
                if waypoint_id in [waypoint_id_list[0], waypoint_id_list[1], waypoint_id_list[-1], waypoint_id_list[-2]]:
                    airlink_resources[flight_direction][waypoint_id] = Airnode(self.env, 
                                                                               airnode_id=waypoint_id,
                                                                               airnode_capacity=1) # Overtake is not allowed
                # Holding unit
                elif waypoint_id == waypoint_id_list[-3]:
                    airlink_resources[flight_direction][waypoint_id] = Airnode(self.env,
                                                                               airnode_id=waypoint_id,
                                                                               airnode_capacity=self.holding_unit_capacity)
                # Cruise segment
                else:
                    airlink_resources[flight_direction][waypoint_id] = Airnode(self.env,
                                                                               airnode_id=waypoint_id,
                                                                               airnode_capacity=self.airlink_capacity)
        return airlink_resources
    
    def get_airspace_states_dict(self):
        return {
            flight_direction: {
                waypoint_id: airnode.get_airnode_states()
                for waypoint_id, airnode in airnodes.items()
            }
            for flight_direction, airnodes in self.airlink_resources.items()
        }
    
    
    def get_airspace_states(self) -> Dict:
        """
        Gets the states of all the airnodes in the airspace.
        """
        all_states = {}
        for flight_direction in self.flight_directions:
            for airnode_id in self.waypoint_ids[flight_direction]:
                all_states |= self.airlink_resources[flight_direction][
                    airnode_id
                ].get_airnode_states()
        return all_states
    
    
    # @staticmethod
    # def calculate_distance(row1, row2, unit='meter') -> float:
    #     return haversine_dist(row1['latitude'], row1['longitude'], 
    #                         row2['latitude'], row2['longitude'], 
    #                         unit=unit)
    
    @staticmethod
    def calculate_distance(row1, row2, unit='meter') -> float:
        return haversine_dist(row1[0], row1[1], 
                            row2[0], row2[1], 
                            unit=unit)    

    @staticmethod
    def calculate_and_store_distances(df) -> Dict:
        distances = defaultdict(dict)
        for (_, row1), (_, row2) in zip(df.iterrows(), df.iloc[1:].iterrows()):
            distance = Airspace.calculate_distance(row1, row2)
            distances[row1['waypoint_id']][row2['waypoint_id']] = distance
            distances[row2['waypoint_id']][row1['waypoint_id']] = distance
        return distances


    @staticmethod
    def calculate_first_last_distance(df, distances) -> Dict:
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        distance = Airspace.calculate_distance(first_row, last_row)
        distances[first_row['waypoint_id']][last_row['waypoint_id']] = distance
        distances[last_row['waypoint_id']][first_row['waypoint_id']] = distance
        return distances


    # @staticmethod
    # def compute_distances_bw_waypoints(df) -> Dict:
    #     # Check if input is a dictionary and convert to DataFrame if necessary
    #     if isinstance(df, dict):
    #         df = pd.DataFrame.from_dict(df, orient='index', columns=['latitude', 'longitude', 'altitude'])
    #         df.index.set_names(['waypoint_id'], inplace=True)
    #         df.reset_index(inplace=True)                
    #     distances = Airspace.calculate_and_store_distances(df)
    #     distances = Airspace.calculate_first_last_distance(df, distances)
    #     return distances
    
    @staticmethod
    def compute_distances_bw_waypoints(waypoint_ids, waypoint_locations) -> Dict:
        waypoint_distances = defaultdict(dict)
        
        for flight_direction, waypoints in waypoint_ids.items():
            for i in range(len(waypoints) - 1):
                waypoint1, waypoint2 = waypoints[i], waypoints[i + 1]
                location1, location2 = waypoint_locations[waypoint1], waypoint_locations[waypoint2]
                distance = Airspace.calculate_distance(location1, location2)
                waypoint_distances[waypoint1][waypoint2] = distance
                waypoint_distances[waypoint2][waypoint1] = distance  # If distance is bidirectional

        return waypoint_distances    

    def get_wind_speed_and_direction(self):
        # Randomly pick a row from the wind data
        wind_data_row = self.wind_data.sample(n=1).iloc[0]
        wind_speed = wind_data_row['windspeed']
        wind_direction = wind_data_row['winddir']
        return wind_speed, wind_direction

    def get_waypoint_rank(self, waypoint_id: str, flight_direction: str) -> int:
        """
        Returns the rank of the waypoint in the flight direction.
        """
        return self.waypoint_ids[flight_direction].index(waypoint_id)
    
    def get_flight_length(self, flight_direction: str) -> int:
        """
        Returns the number of waypoints in the flight direction.
        """
        return len(self.waypoint_ids[flight_direction])

class Airnode:
    def __init__(self, env: simpy.Environment(), 
                 airnode_id: str,
                 airnode_capacity: int = 1,
                 airnode_resource: simpy.Resource = None) -> None:
        self.env = env
        self.airnode_id = airnode_id
        self.airnode_capacity = airnode_capacity
        self.airnode_resource = airnode_resource or simpy.Resource(self.env, capacity=self.airnode_capacity)
    

    def get_airnode_states(self) -> Dict:
        return {
            self.airnode_id: {
                'num_jobs_in_service': self.airnode_resource.count,
                'num_jobs_in_queue': len(self.airnode_resource.queue)
            }
        }
    
    
    def __repr__(self) -> str:
        return f"Airnode: {self.airnode_id}"