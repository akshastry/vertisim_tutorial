from typing import Dict
from ..utils.units import miles_to_m


def time_to_arrival_estimator(aircraft: object) -> float:
    """
    Estimates the time to arrival at the destination vertiport.
    """
    if aircraft.destination_vertiport_id is None:
        return 0
    if aircraft.origin_vertiport_id == aircraft.destination_vertiport_id:
        return 0
    # If the location is a parking pad location, then aircraft is at the vertiport. The time to arrival is zero.
    for vertiport_id, vertiport in aircraft.system_manager.vertiports.items():
        if vertiport.is_node_vertiport_location(aircraft.location):
            return 0
        
    average_flight_time = aircraft.system_manager.get_average_flight_time(aircraft.flight_direction)
    initialization_checker = average_flight_time is None

    # If average flight time is not set, compute and update it.
    if average_flight_time is None:
        average_flight_time = aircraft.system_manager.initial_flight_duration_estimate(aircraft.origin_vertiport_id, 
                                                                                       aircraft.destination_vertiport_id, 
                                                                                       aircraft.aircraft_params['cruise_speed'])
        aircraft.system_manager.set_average_flight_time(aircraft.flight_direction, average_flight_time)
        aircraft.system_manager.increase_flight_count(aircraft.flight_direction)

    if aircraft.location is None:
        attributes = vars(aircraft)
        for attribute, value in attributes.items():
            print(f"{attribute}: {value}")
        print("Time: ", aircraft.env.now)
    waypoint_rank = aircraft.system_manager.airspace.get_waypoint_rank(aircraft.location, aircraft.flight_direction)
    flight_length = aircraft.system_manager.airspace.get_flight_length(aircraft.flight_direction)

    # Calculate estimated time to arrival.
    estimated_time_to_arrival = average_flight_time * (flight_length - waypoint_rank) / flight_length

    # Reset average flight time to None if it was originally None.
    if initialization_checker:
        aircraft.system_manager.set_average_flight_time(aircraft.flight_direction, None)
        aircraft.system_manager.decrease_flight_count(aircraft.flight_direction)

    return estimated_time_to_arrival
