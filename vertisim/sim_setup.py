import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time

# from utils.defaults import sim_params, network_and_demand, vertiport_params, aircraft_params, output_params
import simpy
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict
from .aircraft.aircraft_arrival_setup import AircraftArrivalSetup
from .event_saver import EventSaver
from .system_managers.rl_system_manager import RLSystemManager
from .system_managers.offline_opt_system_manager import OfflineOptimizationSystemManager
from .system_managers.ondemand_system_manager import OnDemandSystemManager
from .aircraft.charging_strategies.ondemand_charging_strategy import OnDemandChargingStrategy
from .aircraft.charging_strategies.rl_charging_strategy import RLChargingStrategy
from .aircraft.charging_strategies.offline_opt_charging_strategy import OfflineOptimizationChargingStrategy
from .module_setup import vertiport_config, create_vertiport_layouts, set_taxi_operations_config, create_airspace_layout
from .passenger_arrival_setup import PassengerArrivalSetup
from .performance_metrics import PerformanceMetrics
from .save_output import save_output
from .scheduler import Scheduler
from .wind.wind import Wind
from .utils.units import mph_to_metersec, degrees_to_radians
from .logger import Logger
from .run_step_by_step_simulation import run_step_by_step_simulation
from .utils.create_output_folders import create_output_folders
from .utils.get_aircraft_count import get_aircraft_count
from .utils.calc_performance_metrics import calculate_passenger_trip_time_stats, calculate_passenger_waiting_time_stats
from .utils.bundle_passenger_related_distributions import bundle_distributions
from .utils.calc_vertiport_distances_from_lat_lng import calc_vertiport_distances
from .utils.helpers import set_seed, get_random_id, create_dict_for_fixes, is_none
from .utils.import_aircraft_energy_consumption_data import import_aircraft_energy_consumption_data
from .utils.setup_distributions import setup_passenger_distributions
from .utils.units import sec_to_ms, ms_to_min



class SimSetup:
    def __init__(self,
                 env: simpy.Environment,
                 sim_params: Dict,
                 sim_mode: Dict,
                 external_optimization_params: Dict,
                 network_and_demand_params: Dict,
                 airspace_params: Dict,
                 passenger_params: Dict,
                 aircraft_params: Dict,
                 output_params: Dict,
                 stopping_events: Dict = None,
                 truncation_events: Dict = None,
                 truncation_event: simpy.Event = None):
        self.sim_params = sim_params
        self.sim_mode = sim_mode
        self.external_optimization_params = external_optimization_params
        self.network_and_demand_params = network_and_demand_params
        self.airspace_params = airspace_params
        self.passenger_params = passenger_params
        self.aircraft_params = aircraft_params
        self.output_params = output_params
        self.truncation_events = truncation_events
        self.truncation_event = truncation_event

        self.sim_start_timestamp = sec_to_ms(1686873600)  # Friday, June 16, 2023 12:00:00 AM
        set_seed(sim_params['random_seed'])
        # Get a random id for the simulation
        self.sim_id = get_random_id()
        self.sim_params['sim_id'] = self.sim_id        
        self.env = env
        self.output_folder_path = create_output_folders(self.output_params['output_folder_path'])
        self.logger, self.aircraft_logger, self.passenger_logger, self.vertiport_logger = self.create_loggers()
        self.stopping_events = stopping_events
        self.vertiport_layouts = self.create_vertiport_layout()      
        self.vertiport_ids = self.get_vertiport_ids() # List of vertiport ids 
        self.vertiport_index_to_id_map = SimSetup.create_vertiport_id_index_mapping(self.vertiport_ids)    
        self.vertiport_id_to_index_map = {v: k for k, v in self.vertiport_index_to_id_map.items()}   
        self.vertiport_distances = self.compute_vertiport_distances() # pd.DataFrame of vertiport connectivity, locations and distances
        self.check_inputs()
        self.vertiports = self.create_vertiports() # Dictionary of vertiport entities

        self.airspace = self.create_airspace()  
        self.wind = self.create_wind()    
        self.node_locations = self.build_node_locations_dict()
        self.structural_entity_groups = self.merge_structural_entity_groups()
        self.event_saver = self.create_event_saver()
        self.taxi_config = self.set_taxi_operations_config()
        # self.aircraft_energy_consumption_data = self.set_aircraft_energy_consumption_data() # Dictionary of aircraft energy consumption data
        self.passenger_distributions = self.create_passenger_distributions()
        
        self.create_training_db_path()
        self.scheduler = self.create_scheduler()
        self.system_manager = self.create_system_manager()

        self.create_aircraft_arrival_process()
        self.create_passenger_arrival_process()


    def create_loggers(self):
        """
        Sets the loggers
        """
        logger = Logger(output_folder_path=self.output_folder_path)
        logger.create_logger(name='main', 
                             env=self.env, 
                             enable_logging=self.sim_params['logging'],
                             level=self.sim_params['log_level'])

        aircraft_logger = Logger(output_folder_path=self.output_folder_path)
        aircraft_logger.create_logger(name='aircraft', 
                                      env=self.env, 
                                      enable_logging=self.sim_params['logging'],
                                      level=self.sim_params['log_level'])

        passenger_logger = Logger(output_folder_path=self.output_folder_path)
        passenger_logger.create_logger(name='passenger', 
                                       env=self.env, 
                                       enable_logging=self.sim_params['logging'],
                                       level=self.sim_params['log_level'])

        vertiport_logger = Logger(output_folder_path=self.output_folder_path)
        vertiport_logger.create_logger(name='vertiport', 
                                       env=self.env, 
                                       enable_logging=self.sim_params['logging'],
                                       level=self.sim_params['log_level'])

        return logger, aircraft_logger, passenger_logger, vertiport_logger
        
    def create_training_db_path(self):
        if self.sim_params['training_data_collection']:
            self.output_params['state_trajectory_db_path'] = os.path.join(self.output_folder_path, self.output_params['state_trajectory_db_name'])
        else:
            self.output_params['state_trajectory_db_path'] = None

    def create_vertiport_layout(self):
        """
        Creates the vertiport layouts and adds them to the network_and_demand dictionary
        """
        # Create vertiport layouts
        vertiport_layouts = create_vertiport_layouts(network_and_demand=self.network_and_demand_params,
                                                     output_folder_path=self.output_folder_path)
        if self.sim_params['verbose']:
            print("Success: Vertiport layout(s) created.")
        return vertiport_layouts
    
    def get_vertiport_ids(self) -> List:
        """
        Returns the vertiport ids
        """
        return list(self.network_and_demand_params['vertiports'].keys()) 

    @staticmethod
    def create_vertiport_id_index_mapping(vertiport_ids: List):
        """
        Creates a mapping between vertiport ids and their index in the list
        """
        return dict(enumerate(vertiport_ids))
    
    def create_wind(self):
        """
        Creates the wind module
        """
        return Wind(static_wind=self.airspace_params['static_wind'],
                    wind_magnitude=mph_to_metersec(self.airspace_params['wind_magnitude_mph']),
                    wind_angle=degrees_to_radians(self.airspace_params['wind_angle_degrees']),
                    wind_data_file_path=self.airspace_params['wind_data_file_path'])
    
    def compute_vertiport_distances(self) -> pd.DataFrame:
        """
        Imports the vertiport distance
        """
        # Import vertiport distance
        vertiport_distances = calc_vertiport_distances(
            vertiport_network_file_path=self.network_and_demand_params['vertiport_network_file_path'],
            vertiport_ids=self.vertiport_ids,
            vertiport_layouts=self.vertiport_layouts,
            network_simulation=self.sim_params['network_simulation'])
        if self.sim_params['verbose']:
            print("Success: Vertiport network imported.")
        return vertiport_distances
    
    def num_vertiports(self):
        """
        Returns the number of vertiports
        """
        return len(self.vertiport_ids)

    def check_inputs(self):
        """
        Checks the inputs
        """
        if self.num_vertiports() != len(self.vertiport_layouts.keys()):
            raise ValueError('The number of vertiport ids and layouts do not match. Check the vertiport network or vertiport input configs.')
            
        if self.sim_mode['offline_optimization'] == True and self.sim_mode['client_server'] == True:
            raise ValueError('Both offline and client_server cannot be True at the same time. Check sim_mode.')
        
        if self.sim_mode['offline_optimization'] == True and self.sim_mode['ondemand'] == True:
            raise ValueError('Both offline optimization and ondemand cannot be True at the same time. Check sim_mode.')
        
        if self.sim_mode['client_server'] == True and self.sim_mode['ondemand'] == True:
            raise ValueError('Both online optimization and ondemand cannot be True at the same time. Check sim_mode.')
        
        if self.sim_mode['offline_optimization'] == False and \
            self.sim_mode['client_server'] == False and \
                self.sim_mode['ondemand'] == False and \
                    self.sim_mode['rl'] == False:
            raise ValueError('At least one of offline optimization, online optimization or ondemand must be True. Check sim_mode.')
        
        if self.sim_mode['offline_optimization'] == True and self.external_optimization_params['flight_schedule_file_path'] is None:
            raise ValueError('Offline optimization cannot be run without a flight schedule. Check network_and_demand_params.')
        
        if self.external_optimization_params['charge_assignment_sensitivity'] and self.external_optimization_params['charge_schedule_file_path'] is None:
            raise ValueError('Charge assignment sensitivity is needed to run charge schedule. Check network_and_demand_params and external_optimization_params.')

    def create_vertiports(self) -> Dict:
        # Set vertiport configurations
        return vertiport_config(
            env=self.env,
            vertiport_layouts=self.vertiport_layouts,
            vertiports=self.network_and_demand_params['vertiports'],
            vertiport_ids=self.vertiport_ids,
            network_and_demand=self.network_and_demand_params,
            aircraft_capacity=self.aircraft_params['pax'],
            pax_waiting_time_threshold=self.sim_params['max_passenger_waiting_time']
        )
    
    def create_event_saver(self):
        """
        Creates the event saver
        """
        # Create event saver
        return EventSaver(env=self.env, 
                          vertiports=self.vertiports, 
                          sim_start_timestamp=self.sim_start_timestamp, 
                          sim_params=self.sim_params,
                          node_locations=self.node_locations)    

    def set_taxi_operations_config(self):
        """
        Sets the taxi operations configuration
        """
        # Set taxi operations config
        taxi_config = set_taxi_operations_config(
            is_simultaneous_taxi_and_take_off_allowed=self.sim_params['simultaneous_taxi_and_take_off']
        )
        if self.sim_params['verbose']:
            print("Success: Vertiport configuration is set.")
        return taxi_config

    def set_aircraft_energy_consumption_data(self):
        """
        Sets the aircraft energy consumption data
        """
        if self.sim_params['battery_model_sim']:
            return self.import_aircraft_energy_consumption_data()
        else:
            return None

    def import_aircraft_energy_consumption_data(self):
        """
        Imports the aircraft energy consumption data
        """
        # Import aircraft energy consumption data
        if self.aircraft_params['aircraft_energy_consumption_data_folder_path'] is not None:
            aircraft_energy_consumption_data = import_aircraft_energy_consumption_data(
                folder_path=self.aircraft_params['aircraft_energy_consumption_data_folder_path'],
                aircraft_models=list(self.aircraft_params['aircraft_model']),
                ranges=list(self.vertiport_distances['distance']))
            if self.sim_params['verbose']:
                print("Success: Aircraft energy consumption data imported.")
        else:
            if self.sim_params['verbose']:
                print("Aircraft energy consumption data folder path is not provided.")
            aircraft_energy_consumption_data = None
        return aircraft_energy_consumption_data

    def bundle_passenger_distributions(self):
        # Bundle passenger related distributions
        return bundle_distributions(
            self.passenger_params['car_to_entrance_walking_time_dist'],
            self.passenger_params['security_check_time_dist'],
            self.passenger_params['waiting_room_to_boarding_gate_walking_time_dist'],
            self.passenger_params['boarding_gate_to_aircraft_time_dist'],
            self.passenger_params['deboard_aircraft_and_walk_to_exit_dist'],
        )

    def create_passenger_distributions(self):
        """
        Creates the passenger distribution
        """
        if self.sim_params['only_aircraft_simulation']:
            return None
        passenger_distributions_bundle = self.bundle_passenger_distributions()
        # Setup passenger distributions
        return setup_passenger_distributions(**passenger_distributions_bundle)

    def create_scheduler(self):
        # Create scheduler
        return Scheduler(env=self.env,
                            vertiports=self.vertiports,
                            system_manager=None,
                            aircraft_capacity=self.aircraft_params['pax'],
                            is_fixed_schedule=bool(self.external_optimization_params['flight_schedule_file_path']),
                            charge_assignment_sensitivity=self.external_optimization_params['charge_assignment_sensitivity'],
                            logger=self.logger,
                            aircraft_logger=self.aircraft_logger,
                            passenger_logger=self.passenger_logger,
                            vertiport_logger=self.vertiport_logger)

    def create_airspace(self):
        # Create airspace
        return create_airspace_layout(env=self.env,
                                    vertiports=self.vertiports,
                                    vertiport_distances=self.vertiport_distances,
                                    airspace_params=self.airspace_params)
    
    def build_node_locations_dict(self):
        # Build node locations dictionary
        node_locations = {}
        # Merge all vertiport element locations
        for _, vertiport_config in self.vertiports.items():
            node_locations |= vertiport_config.vertiport_layout.vertiport_element_locations
        # Merge all airspace element locations
        node_locations |= self.airspace.waypoint_locations
        node_locations |= {None: None}
        return node_locations
    
    def merge_structural_entity_groups(self) -> dict:
        merged_dict = defaultdict(list)
        for _, vertiport_config in self.vertiports.items():
            structural_entity_groups = vertiport_config.vertiport_layout.structural_entity_groups
            for entity, value in structural_entity_groups.items():
                merged_dict[entity].extend(value)
        fix_nodes = create_dict_for_fixes(self.airspace.waypoint_ids)
        merged_dict |= fix_nodes
        return dict(merged_dict)
    
    @staticmethod
    def check_system_manager_type(sim_mode):
        if sim_mode['offline_optimization'] and sim_mode['client_server']:
            raise ValueError("Both offline and online optimization cannot be True at the same time. Check sim_mode params")

    def get_system_manager_type(self):
        # Check system manager type input
        SimSetup.check_system_manager_type(self.sim_mode)
        # Return system manager type
        if self.sim_mode['offline_optimization']:
            return 'offline_optimization'
        elif self.sim_mode['client_server'] or self.sim_mode['rl']:
            return 'rl'
        else:
            return 'ondemand'
        
    @staticmethod
    def set_system_manager_type(system_manager_type, **kwargs):
        if system_manager_type == 'offline_optimization':
            return OfflineOptimizationSystemManager(**kwargs)
        elif system_manager_type == 'rl':
            return RLSystemManager(**kwargs)
        else:
            return OnDemandSystemManager(**kwargs) 
    
    def create_system_manager(self):
        system_manager_type = self.get_system_manager_type()
        configurations = {
            'ondemand': {
                'env': self.env,
                'vertiports': self.vertiports,
                'vertiport_ids': self.vertiport_ids,
                'vertiport_id_to_index_map': self.vertiport_id_to_index_map,
                'num_initial_aircraft': get_aircraft_count(self.network_and_demand_params),
                'scheduler': self.scheduler,
                'wind': self.wind,
                'airspace': self.airspace,
                'taxi_config': self.taxi_config,
                'sim_params': self.sim_params,
                'output_params': self.output_params,
                'aircraft_params': self.aircraft_params,
                'vertiport_distances': self.vertiport_distances,
                'passenger_distributions': self.passenger_distributions,
                'event_saver': self.event_saver,
                'node_locations': self.node_locations,
                'logger': self.logger,
                'aircraft_logger': self.aircraft_logger,
                'passenger_logger': self.passenger_logger,
                'sim_mode': system_manager_type
            },
            'rl': {
                'env': self.env,
                'vertiports': self.vertiports,
                'vertiport_ids': self.vertiport_ids,
                'vertiport_id_to_index_map': self.vertiport_id_to_index_map,                
                'num_initial_aircraft': get_aircraft_count(self.network_and_demand_params),
                'scheduler': self.scheduler,
                'wind': self.wind,
                'airspace': self.airspace,
                'taxi_config': self.taxi_config,
                'sim_params': self.sim_params,
                'external_optimization_params': self.external_optimization_params,
                'output_params': self.output_params,
                'aircraft_params': self.aircraft_params,
                'vertiport_distances': self.vertiport_distances,
                'passenger_distributions': self.passenger_distributions,
                'event_saver': self.event_saver,
                'node_locations': self.node_locations,
                'logger': self.logger,
                'aircraft_logger': self.aircraft_logger,
                'passenger_logger': self.passenger_logger,
                'stopping_events': self.stopping_events,
                'truncation_events': self.truncation_events,
                'truncation_event': self.truncation_event,
                'sim_mode': system_manager_type,
                'periodic_stopping': not is_none(self.external_optimization_params['periodic_time_step'])
            },
            'offline_optimization': {
                'env': self.env,
                'vertiports': self.vertiports,
                'vertiport_ids': self.vertiport_ids,
                'vertiport_id_to_index_map': self.vertiport_id_to_index_map,                
                'num_initial_aircraft': get_aircraft_count(self.network_and_demand_params),
                'scheduler': self.scheduler,
                'wind': self.wind,
                'airspace': self.airspace,
                'taxi_config': self.taxi_config,
                'sim_params': self.sim_params,
                'output_params': self.output_params,
                'aircraft_params': self.aircraft_params,
                'vertiport_distances': self.vertiport_distances,
                'passenger_distributions': self.passenger_distributions,
                'event_saver': self.event_saver,
                'node_locations': self.node_locations,
                'logger': self.logger,
                'aircraft_logger': self.aircraft_logger,
                'passenger_logger': self.passenger_logger,
                'sim_mode': system_manager_type,
                'external_optimization_params': self.external_optimization_params
            },
        }

        return SimSetup.set_system_manager_type(system_manager_type,
                                                **configurations[system_manager_type])
    
    def get_charging_strategy(self):
        if self.sim_mode['offline_optimization']:
            return OfflineOptimizationChargingStrategy()
        elif self.sim_mode['rl']:
            return RLChargingStrategy(soc_increment_per_charge_event=self.external_optimization_params['soc_increment_per_charge_event'],
                                      soc_reward_threshold=self.external_optimization_params['reward_function_parameters']['soc_reward_threshold'])
        else:
            return OnDemandChargingStrategy()
        
    def create_aircraft_arrival_process(sim_setup: Dict):
        # Load or generate aircraft arrival schedule and start the aircraft arrival process
        flight_schedule_file_path = sim_setup.external_optimization_params.get('flight_schedule_file_path', None)
        charge_schedule_file_path = sim_setup.external_optimization_params.get('charge_schedule_file_path', None)

        # Load aircraft arrival schedule
        aircraft_arrival_process = AircraftArrivalSetup(
            env=sim_setup.env,
            network_simulation=sim_setup.sim_params['network_simulation'],
            vertiport_configs=sim_setup.network_and_demand_params['vertiports'],
            vertiport_layouts=sim_setup.vertiport_layouts,
            flight_schedule_file_path=flight_schedule_file_path,
            charge_schedule_file_path=charge_schedule_file_path,
            aircraft_params=sim_setup.aircraft_params,
            system_manager=sim_setup.system_manager,
            scheduler=sim_setup.scheduler,
            wind=sim_setup.wind,
            structural_entity_groups=sim_setup.structural_entity_groups,
            event_saver=sim_setup.event_saver,
            logger=sim_setup.logger,
            aircraft_logger=sim_setup.aircraft_logger,
            vertiport_logger=sim_setup.vertiport_logger,
            charging_strategy=sim_setup.get_charging_strategy()
        )

        aircraft_arrival_process.create_aircraft_arrival()

        if sim_setup.sim_params['verbose']:
            print("Success: Aircraft arrival process is created.")

    def create_passenger_arrival_process(sim_setup: Dict):
        if sim_setup.sim_params['only_aircraft_simulation'] is False:
            # Load or generate passenger arrival schedule and start the passenger arrival process
            passenger_arrival_process = PassengerArrivalSetup(
                env=sim_setup.env,
                network_and_demand=sim_setup.network_and_demand_params,
                passenger_params=sim_setup.passenger_params,
                vertiport_ids=sim_setup.vertiport_ids,
                system_manager=sim_setup.system_manager,
                aircraft_params=None,
                network_simulation=sim_setup.sim_params['network_simulation'],
                passenger_logger=sim_setup.passenger_logger
                )
            sim_setup.env.process(
                passenger_arrival_process.create_passenger_arrival()
            )
            if sim_setup.sim_params['verbose']:
                print("Success: Passenger arrival process is created.")                    

    def calculate_performance_metrics(self):
        # Print and save performance metrics
        # print("Calculating performance metrics... \n")
        return PerformanceMetrics(sim_start_timestamp=self.sim_start_timestamp,
                                  aircraft_params=self.aircraft_params,
                                  only_aircraft_simulation=self.sim_params['only_aircraft_simulation'],
                                  is_network_simulation=self.sim_params['network_simulation'],
                                  performance_metric_trackers=self.event_saver.performance_metrics(),
                                  vertiport_ids=self.vertiport_ids,
                                  flight_directions=self.airspace.flight_directions,
                                  sim_id=self.sim_id,
                                  sim_end_timestamp=self.env.now,
                                  client_server_mode=self.sim_mode['rl'],
                                  sim_params=self.sim_params,
                                  output_params=self.output_params,
                                  verbose=self.sim_params['verbose']).performance_metrics
    
    def log_brief_metrics(self, print_metrics=False):
        calculate_passenger_trip_time_stats(passenger_trip_time_tracker=self.event_saver.performance_metrics()['passenger_trip_time_tracker'],
                                            flight_directions=self.airspace.flight_directions,
                                            logger=self.logger,
                                            print_metrics=print_metrics)
        calculate_passenger_waiting_time_stats(passenger_waiting_time_tracker=self.event_saver.performance_metrics()['passenger_waiting_time'],
                                               logger=self.logger,
                                               print_metrics=print_metrics)
                    

    def save_results(self, performance_metrics):
        # Save the results
        save_output(simulation_params={
                        'sim_params': self.sim_params,
                        'network_and_demand': self.network_and_demand_params,
                        'passenger_params': self.passenger_params,
                        'aircraft_params': self.aircraft_params,
                        'airspace_params': self.airspace_params,
                        'output_params': self.output_params
                    },
                    trajectories={
                        'aircraft_agent_trajectory': self.event_saver.aircraft_agent_trajectory,
                        'passenger_agent_trajectory': self.event_saver.passenger_agent_trajectory                        
                    },
                    performance_metrics=performance_metrics,
                    simulationID=self.sim_id,
                    flight_directions=self.airspace.flight_directions
                )
        # print(f"Success: Results are saved in {output_path}.")



