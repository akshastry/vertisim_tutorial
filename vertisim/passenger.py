import random
from .utils.distribution_generator import DistributionGenerator
from .utils.units import sec_to_ms
from typing import Dict, Union, Any


class Passenger:
    """
    Passenger agent. If the constant is None, then the distributions from system_manager will be used.
    Distributions are not inputted directly to the passenger, because it will increase the overhead of each passenger
    object.
    """

    def __init__(self, env, passenger_id,
                 origin_vertiport_id: Any,
                 destination_vertiport_id: Any,
                 location: str,
                 passenger_params: Dict,
                 system_manager: object):
        self.env = env
        self.passenger_id = passenger_id
        self.origin_vertiport_id = origin_vertiport_id
        self.destination_vertiport_id = destination_vertiport_id
        self.location = location
        self.passenger_params = passenger_params
        self.car_to_entrance_walking_time_constant = sec_to_ms(self.passenger_params['car_to_entrance_walking_time_constant'])
        self.security_check_time_constant = sec_to_ms(self.passenger_params['security_check_time_constant'])
        self.waiting_room_to_boarding_gate_walking_time_constant = sec_to_ms(self.passenger_params['waiting_room_to_boarding_gate_walking_time_constant'])
        self.boarding_gate_to_aircraft_time_constant = sec_to_ms(self.passenger_params['boarding_gate_to_aircraft_time_constant'])
        self.deboard_aircraft_and_walk_to_exit_constant = sec_to_ms(self.passenger_params['deboard_aircraft_and_walk_to_exit_constant'])
        self.randomize_constants = self.passenger_params['randomize_constants']
        self.system_manager = system_manager
        self.vertiport_arrival_time = None
        self.waiting_room_arrival_time = None
        self.waiting_room_store = None
        self.flight_queue_store = None
        self.flight_assignment_time = None
        self.departure_time = None

    def walk_to_entrance(self):
        """ From car to the exit gate."""
        if self.car_to_entrance_walking_time_constant is not None:
            if self.randomize_constants:
                yield self.env.timeout(random.randint(
                    max(0, round(
                        self.car_to_entrance_walking_time_constant - self.car_to_entrance_walking_time_constant / 2)),
                    round(self.car_to_entrance_walking_time_constant + self.car_to_entrance_walking_time_constant / 2)
                )
                )
            else:
                yield self.env.timeout(self.car_to_entrance_walking_time_constant)
        elif self.system_manager.passenger_distributions['car_to_entrance_walking_time_dist']:
            dist = self.system_manager.passenger_distributions['car_to_entrance_walking_time_dist']
            distribution = DistributionGenerator(
                distribution_params=dist,
                max_val_in_dist=dist['max_val_in_dist'])
            yield self.env.timeout(distribution.pick_number_from_distribution())
        else:
            raise ValueError("No walking time distribution or constant defined.")

    def security_check(self):
        """ Security check."""
        if self.security_check_time_constant is not None:
            if self.randomize_constants:
                yield self.env.timeout(random.randint(
                    max(0, round(self.security_check_time_constant - self.security_check_time_constant / 2)),
                    round(self.security_check_time_constant + self.security_check_time_constant / 2)
                )
                )
            else:
                yield self.env.timeout(self.security_check_time_constant)
        elif self.system_manager.passenger_distributions['security_check_time_dist']:
            dist = self.system_manager.passenger_distributions['security_check_time_dist']
            distribution = DistributionGenerator(distribution_params=dist,
                                                 max_val_in_dist=dist['max_val_in_dist'])
            yield self.env.timeout(distribution.pick_number_from_distribution())
        else:
            raise ValueError("No security check time distribution or constant defined.")

    def waiting_room_to_boarding_gate(self):
        """ From waiting room to boarding gate."""
        if self.waiting_room_to_boarding_gate_walking_time_constant is not None:
            if self.randomize_constants:
                yield self.env.timeout(random.randint(
                    max(0, round(
                        self.waiting_room_to_boarding_gate_walking_time_constant - self.waiting_room_to_boarding_gate_walking_time_constant / 2)),
                    round(
                        self.waiting_room_to_boarding_gate_walking_time_constant + self.waiting_room_to_boarding_gate_walking_time_constant / 2)
                )
                )
            else:
                yield self.env.timeout(self.waiting_room_to_boarding_gate_walking_time_constant)
        elif self.system_manager.passenger_distributions['waiting_room_to_boarding_gate_walking_time_dist']:
            dist = self.system_manager.passenger_distributions['waiting_room_to_boarding_gate_walking_time_dist']
            distribution = DistributionGenerator(
                distribution_params=dist,
                max_val_in_dist=dist['max_val_in_dist'])
            yield self.env.timeout(distribution.pick_number_from_distribution())
        else:
            raise ValueError("No waiting room to boarding gate walking time distribution or constant defined.")

    def boarding_gate_to_aircraft(self):
        """ From boarding gate to aircraft."""
        if self.boarding_gate_to_aircraft_time_constant is not None:
            if self.randomize_constants:
                yield self.env.timeout(random.randint(
                    max(0, round(
                        self.boarding_gate_to_aircraft_time_constant - self.boarding_gate_to_aircraft_time_constant / 2)),
                    round(
                        self.boarding_gate_to_aircraft_time_constant + self.boarding_gate_to_aircraft_time_constant / 2)
                )
                )
            else:
                yield self.env.timeout(self.boarding_gate_to_aircraft_time_constant)
        elif self.system_manager.passenger_distributions['boarding_gate_to_aircraft_time_dist']:
            dist = self.system_manager.passenger_distributions['boarding_gate_to_aircraft_time_dist']
            distribution = DistributionGenerator(
                distribution_params=dist,
                max_val_in_dist=dist['max_val_in_dist'])
            yield self.env.timeout(distribution.pick_number_from_distribution())
        else:
            raise ValueError("No boarding gate to aircraft time distribution or constant defined.")
