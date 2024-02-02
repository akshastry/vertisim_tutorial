from .vertiport_layout_creator import VertiportLayout
from .configurator import VertiportConfig, TaxiOperationsConfig
from .airspace_creator import AirspaceCreator
from .airspace import Airspace
from collections import defaultdict
from typing import Dict, List


def create_vertiport_layouts(network_and_demand: Dict, output_folder_path: str) -> Dict:
    layout_file_path = network_and_demand['vertiport_layout_file_path']
    vertiport_layouts = defaultdict(dict)
    for vertiport_id, vertiport_data in network_and_demand['vertiports'].items():
        layout = vertiport_data['layout']
        vertiport_layout = VertiportLayout(file_path=layout_file_path, 
                                           layout_type=layout,
                                           vertiport_id=vertiport_id,
                                           output_folder_path=output_folder_path)
        vertiport_layouts[vertiport_id] = vertiport_layout
    return vertiport_layouts

def create_airspace_layout(env: object, 
                            vertiports: Dict, 
                            vertiport_distances, 
                            airspace_params) -> AirspaceCreator:
    # Create the airspace layout
    airspace_creator = AirspaceCreator(vertiports=vertiports,
                                       airspace_params=airspace_params,
                                       vertiport_distances=vertiport_distances,
                                       cruise_altitude=airspace_params['cruise_altitude'])
    return Airspace(env=env,
                    holding_unit_capacity=airspace_params['holding_unit_capacity'],
                    airlink_capacity=airspace_params['airlink_capacity'],
                    airlink_segment_length_mile_miles=airspace_params['airlink_segment_length_mile'],
                    waypoint_ids=airspace_creator.waypoint_ids,
                    waypoint_locations=airspace_creator.waypoint_locations,
                    flight_directions=airspace_creator.flight_directions)


def vertiport_config(env: object,
                     vertiports: dict,
                     vertiport_layouts: dict,
                     vertiport_ids: list,
                     network_and_demand: dict,
                     aircraft_capacity: int,
                     pax_waiting_time_threshold: int) -> dict:
    
    return {vertiport_id: VertiportConfig(env=env,
                                          vertiport_id=vertiport_id,
                                          vertiport_data=vertiport_data,
                                          vertiport_layout=vertiport_layouts[vertiport_id],
                                          vertiport_ids=vertiport_ids,
                                          network_and_demand=network_and_demand,
                                          aircraft_capacity=aircraft_capacity,
                                          pax_waiting_time_threshold=pax_waiting_time_threshold)
                for vertiport_id, vertiport_data in vertiports.items()
            }

def set_taxi_operations_config(is_simultaneous_taxi_and_take_off_allowed) -> TaxiOperationsConfig:
    TaxiOperationsConfig.is_simultaneous_taxi_and_take_off_allowed = is_simultaneous_taxi_and_take_off_allowed
    return TaxiOperationsConfig