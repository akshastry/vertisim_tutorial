from typing import Dict, Union, List, Any
import pandas as pd
from .utils.read_files import read_input_file
from .utils.helpers import compute_interarrival_times_from_schedule, miliseconds_to_hms, pick_random_file
from .utils.generate_artificial_supply_demand import generate_passenger_network_demand
from .utils.units import sec_to_ms
from .passenger import Passenger


class PassengerArrivalSetup:
    def __init__(self,
                 env: object,
                 network_and_demand: Dict,
                 passenger_params: Dict,
                 vertiport_ids: List,
                 system_manager: int,
                 aircraft_params: Dict = None,
                 network_simulation: bool = False,
                 passenger_logger: object = None):
        self.env = env
        self.network_and_demand = network_and_demand
        self.vertiport_configs = network_and_demand['vertiports']
        self.passenger_params = passenger_params
        self.vertiport_ids = vertiport_ids
        self.system_manager = system_manager
        self.aircraft_params = aircraft_params
        self.network_simulation = network_simulation
        self.passenger_logger = passenger_logger

    @staticmethod
    def check_passenger_arrival_schedule(passenger_arrival_schedule: pd.DataFrame):
        # Check if passenger ID column exists.
        if 'passenger_id' not in passenger_arrival_schedule.columns:
            raise ValueError('Passenger arrival schedule must have a column named passenger_id.')
        # Check if passenger arrival time column exists.
        if 'passenger_arrival_time' not in passenger_arrival_schedule.columns:
            raise ValueError('Passenger arrival schedule must have a column named passenger_arrival_time.')
        # Check if arrival vertiport id column exists.
        if 'origin_vertiport_id' not in passenger_arrival_schedule.columns:
            raise ValueError('Passenger arrival schedule must have a column named origin_vertiport_id.')
        # Check if passenger destination vertiport ID column exists.
        if 'destination_vertiport_id' not in passenger_arrival_schedule.columns:
            raise ValueError('Passenger arrival schedule must have a column named destination_vertiport_id.')
        # Check if passenger interarrival time column exists.
        if 'interarrival_time' not in passenger_arrival_schedule.columns:
            passenger_arrival_schedule['interarrival_time'] = compute_interarrival_times_from_schedule(passenger_arrival_schedule['passenger_arrival_time'])

    def setup_passenger_arrival(self):
        if self.network_and_demand['passenger_schedule_file_path'] is not None:
            # Load passenger arrival schedule from the path provided on self.passenger_arrival_schedule.
            passenger_arrival_schedule = read_input_file(self.network_and_demand['passenger_schedule_file_path'])
            # Sort the passenger arrival schedule by passenger_arrival_time
            passenger_arrival_schedule.sort_values(by='passenger_arrival_time', inplace=True)
            passenger_arrival_schedule.reset_index(drop=True, inplace=True)
            # Check if the columns of the input file are correct.
            self.check_passenger_arrival_schedule(passenger_arrival_schedule)
            # print("Success: Passenger arrival schedule loaded from file.")
        
        elif self.network_and_demand['autoregressive_demand_files_path'] is not None:
            # Pick a random file from the list of autoregressive demand files.
            file_name = pick_random_file(self.network_and_demand['autoregressive_demand_files_path'])
            # Load passenger arrival schedule from the path
            passenger_arrival_schedule = read_input_file(f"{self.network_and_demand['autoregressive_demand_files_path']}/{file_name}")
            # Check if the columns of the input file are correct.
            self.check_passenger_arrival_schedule(passenger_arrival_schedule)        
        else:
            passenger_arrival_rates = read_input_file(self.network_and_demand['passenger_arrival_rates_path'])
            # Generate artificial passenger arrival schedule.
            passenger_arrival_schedule = generate_passenger_network_demand(
                vertiport_configs=self.vertiport_configs,
                vertiport_ids=self.vertiport_ids,
                demand_probabilities=self.network_and_demand['demand_probabilities'],
                passenger_arrival_rates=passenger_arrival_rates,
                network_simulation=self.network_simulation
            )
            # print("Success: Passenger arrival schedule is generated from the given distribution.")
        
        self.load_pax_arrival_df_to_vertiports(passenger_arrival_schedule)
        return passenger_arrival_schedule

    def load_pax_arrival_df_to_vertiports(self, passenger_arrival_schedule):
        for vertiport_id in self.vertiport_ids:
            self.system_manager.vertiports[vertiport_id].pax_arrival_df = passenger_arrival_schedule

    def create_passenger_arrival(self):
        # Setup passenger arrival schedule.
        passenger_arrival_schedule = self.setup_passenger_arrival()
        self.system_manager.total_demand = len(passenger_arrival_schedule)
        self.system_manager.logger.warning(f"Total number of passengers: {self.system_manager.total_demand}")

        if not self.network_simulation:
            if self.aircraft_params['time_charging']:
                charging_time = self.aircraft_params['time_charging']
            else:
                charging_time = self.aircraft_params['charging_time_dist']['parameters']['scale']

            origin_vertiport_id = list(self.vertiport_configs.keys())[0]
            # Time to fill the vertiport with aircraft. This number is inputed in sim_params['initial_parking_occupancy']
            if self.vertiport_configs[origin_vertiport_id]['passenger_arrival_process']['passenger_interarrival_constant'] is not None:
                passenger_arrival_interval_time = self.vertiport_configs[origin_vertiport_id]['passenger_arrival_process']['passenger_interarrival_constant']
            else:
                passenger_arrival_interval_time = self.vertiport_configs[origin_vertiport_id]['passenger_arrival_process']['passenger_arrival_distribution']['parameters']['scale']   
            num_parking_pads = self.system_manager.vertiports[origin_vertiport_id].vertiport_layout.num_parking_pad # This is the number of parking pads at the first vertiport. Might be improved.
            time_to_fill_vertiport = int(num_parking_pads/2) * (
                self.aircraft_params['time_descend_transition'] +
                self.aircraft_params['time_hover_descend'] +
                self.aircraft_params['time_rotor_spin_down'] +
                22) + charging_time - self.aircraft_params['pax']/2 * passenger_arrival_interval_time
            time_to_fill_vertiport = max(0, time_to_fill_vertiport)

            self.passenger_logger.info(f"Single vertiport simulation initiated. Skipping time duration of {time_to_fill_vertiport} seconds to fill vertiport with aircraft.")

            yield self.env.timeout(sec_to_ms(time_to_fill_vertiport))

        # Create passenger arrival process.
        for passenger_id in passenger_arrival_schedule.passenger_id:
            # Get the passenger's interarrival time
            interarrival_time = passenger_arrival_schedule.loc[passenger_arrival_schedule.passenger_id == passenger_id, 'interarrival_time'].values[0]
            # Get the passenger's origin vertiport id
            origin_vertiport_id = passenger_arrival_schedule.loc[passenger_arrival_schedule.passenger_id == passenger_id, 'origin_vertiport_id'].values[0]
            # Get the passenger's destination vertiport id
            destination_vertiport_id = passenger_arrival_schedule.loc[passenger_arrival_schedule.passenger_id == passenger_id, 'destination_vertiport_id'].values[0]
            # Get the location of the passenger
            passenger_location = f'{origin_vertiport_id}_ENTRANCE'

            yield self.env.timeout(sec_to_ms(interarrival_time))
            self.passenger_logger.info(f'Passenger {passenger_id} arrives at {origin_vertiport_id} to fly to {destination_vertiport_id}')
            # Simulate passenger.
            self.initiate_passenger_arrival(
                    passenger_id=passenger_id,
                    location=passenger_location,
                    origin_vertiport_id=origin_vertiport_id,
                    destination_vertiport_id=destination_vertiport_id
            )
        
        # print(f"All {len(passenger_arrival_schedule)} passengers have arrived at time (sec) {miliseconds_to_hms(self.env.now)}.")
        self.system_manager.passenger_arrival_complete = True
        # print(f"Total number of waiting passengers: {self.system_manager.get_total_waiting_passenger_count()}")

    def initiate_passenger_arrival(self, 
                                   passenger_id: int,
                                   location: str,
                                   origin_vertiport_id: Any,
                                   destination_vertiport_id: Any):
        # Create a passenger object
        passenger = Passenger(
            env=self.env,
            passenger_id=passenger_id,
            location=location,
            origin_vertiport_id=origin_vertiport_id,
            destination_vertiport_id=destination_vertiport_id,
            passenger_params=self.passenger_params,
            system_manager=self.system_manager
        )

        # Add passenger to the passenger agents dictionary.
        self.system_manager.passenger_agents[f'passenger_{passenger_id}'] = passenger
        # Initiate passenger arrival.
        self.env.process(self.system_manager.simulate_passenger_arrival(passenger))
