"""
There are three ways to set up aircraft arrival (initiation) process for network simulation:
1. Define the arrival process in the config file. It will generate an artificial schedule
2. Define aircraft arrival/departure schedule in a csv file. It will read the schedule 
    and create aircraft departure processes based on the schedule.
3. Optimization output simulation: Define aircraft departure schedule and aircraft charging schedule.
    The charge schedule should include the initial idle aircraft as well.
"""


import pandas as pd

from .aircraft import Aircraft, AircraftStatus
import pandas as pd
from ..utils.units import sec_to_ms, ms_to_sec
from ..utils.helpers import compute_interarrival_times_from_schedule, check_if_none_exists, \
    careful_conversion, lower_str
from ..utils.generate_artificial_supply_demand import generate_aircraft_network_schedule
# from utils.defaults import DEFAULT_AIRCRAFT_ARRIVAL_PROCESS
from typing import Dict, List, Union, Tuple
from ..utils.read_files import read_input_file


class AircraftArrivalSetup:
    def __init__(self,
                 env: object,
                 network_simulation: bool,
                 vertiport_layouts: Dict,
                 vertiport_configs: Union[Dict, None],
                 flight_schedule_file_path: Union[str, None],
                 charge_schedule_file_path: Union[str, None],
                 aircraft_params: Dict,
                 system_manager: int,
                 scheduler: object,
                 wind: object,
                 structural_entity_groups: Dict,                
                 event_saver: object,
                 logger: object,
                 aircraft_logger: object,
                 vertiport_logger: object,
                 charging_strategy: object):
        self.env = env
        self.network_simulation = network_simulation
        self.vertiport_layouts = vertiport_layouts
        self.flight_schedule_file_path = flight_schedule_file_path
        self.charge_schedule_file_path = charge_schedule_file_path
        self.vertiport_configs = vertiport_configs
        self.aircraft_params = aircraft_params
        self.system_manager = system_manager
        self.scheduler = scheduler
        self.wind = wind
        self.structural_entity_groups = structural_entity_groups
        self.event_saver = event_saver
        self.num_aircraft = None
        # logfile = set_logfile_path(log_file_name='main', output_folder_path=output_folder_path)
        # self.logger = setup_logger(name='main', log_file=logfile, env=self.env)
        self.logger = logger
        self.aircraft_logger = aircraft_logger
        self.vertiport_logger = vertiport_logger
        self.charging_strategy = charging_strategy
            
    @staticmethod
    def validate_columns(dataframe: pd.DataFrame, required_columns: list, optional_columns: dict, check_none_columns: list) -> None:
        """
        Validate required columns, optional columns and columns to check for None values.
        :param dataframe: DataFrame to be validated.
        :param required_columns: A list of required column names.
        :param optional_columns: A dictionary of optional column names and their default values.
        :param check_none_columns: A list of column names to check for None values.
        :return: dataframe.
        """
        for column in required_columns:
            # Check if the column exists
            if column not in dataframe.columns:
                raise ValueError(f'{column} column does not exist in the aircraft arrival process file.')
        
        for column in check_none_columns:
            # Check for None values in the column
            if check_if_none_exists(dataframe[column]):
                raise ValueError(f'{column} column has None value.')
        
        for column, default_value in optional_columns.items():
            if column not in dataframe.columns:
                dataframe[column] = default_value
        return dataframe
                # if column not in ['interarrival_time', 'passengers_onboard']:
                #     print(
                #         f"{column} is not defined in the input file. The default value of {default_value} is used. If you "
                #         "want to define the initial value for each aircraft type, please add the column "
                #         f"'{column}' to the input file."
                #     )

    # ---------------------------- Setup aircraft flight schedule ---------------------------- #
    def check_aircraft_schedule(self, aircraft_schedule: pd.DataFrame):
        """
        Check if the columns of the input file are correct.
        :param aircraft_schedule:
        :return:
        """
        if not self.charge_schedule_file_path:
            required_columns = [
                'aircraft_arrival_time',
                'aircraft_pushback_time',
                'tail_number',
                'origin_vertiport_id',
                'destination_vertiport_id',
                'location',
                'process'
                # 'passengers_onboard'
            ]
        
            check_none_columns = [
                'aircraft_arrival_time',
                'aircraft_pushback_time',
                'tail_number',
                'origin_vertiport_id',
                'destination_vertiport_id',
                # 'passengers_onboard'
            ]
        
            optional_columns = {
                # Column name: default value
                'serviced_time_at_the_location': 0,
                'soc': self.aircraft_params['max_init_soc'],
                'priority': 1,
                'passengers_onboard': [[] for _ in range(len(aircraft_schedule))],
                # 'interarrival_time': compute_interarrival_times_from_schedule(aircraft_schedule['aircraft_arrival_time']),
                'pushback_interarrival_time': compute_interarrival_times_from_schedule(aircraft_schedule['aircraft_arrival_time'])
            }
        else:
            required_columns = [
                'aircraft_pushback_time',
                'origin_vertiport_id',
                'destination_vertiport_id',
                'soc'
            ]
        
            check_none_columns = [
                'aircraft_pushback_time',
                'origin_vertiport_id',
                'destination_vertiport_id',
                'soc'
            ]
        
            optional_columns = {'pushback_interarrival_time': compute_interarrival_times_from_schedule(aircraft_schedule['aircraft_pushback_time'])}
        return AircraftArrivalSetup.validate_columns(aircraft_schedule, required_columns, optional_columns, check_none_columns)

        
    def setup_aircraft_schedule(self) -> pd.DataFrame:
        if self.flight_schedule_file_path is not None:
            return self._load_schedule_from_file()
        else:
            return self._generate_artificial_schedule()

    def _load_schedule_from_file(self):
        # Load and process aircraft arrival schedule
        schedule, initial_states = self._read_and_process_schedule(self.flight_schedule_file_path)

        # Set up scheduler with the processed schedule
        self._setup_scheduler_with_schedule(schedule)

        print("Success: Aircraft arrival schedule loaded from file.")
        return initial_states

    def _generate_artificial_schedule(self):
        # Generate artificial aircraft arrival schedule
        schedule = generate_aircraft_network_schedule(
            vertiport_configs=self.vertiport_configs,
            vertiport_layouts=self.vertiport_layouts,
            aircraft_params=self.aircraft_params,
            network_simulation=self.network_simulation)
            
        # Get the number of aircraft
        self.num_aircraft = len(schedule)
        return schedule

    def _read_and_process_schedule(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load aircraft arrival schedule from the path and sort by arrival time
        schedule = read_input_file(path).sort_values(by='aircraft_pushback_time')

        # Check if the columns of the input file are correct
        schedule = self.check_aircraft_schedule(schedule)

        # Convert the columns to milliseconds
        schedule = AircraftArrivalSetup.convert_schedule_columns_to_ms(schedule, 'flight_schedule')

        # Get the initial state of the system from the input file
        if self.charge_schedule_file_path:
            initial_states, initial_states_indices = AircraftArrivalSetup._get_initial_system_state_from_optimization(
                schedule=schedule,
                operation='flight_schedule')
            schedule = schedule.drop(initial_states_indices)
        else:
            initial_states = AircraftArrivalSetup._get_initial_system_state(schedule)
        return schedule, initial_states

    @staticmethod
    def _get_initial_system_state(schedule: pd.DataFrame) -> pd.DataFrame:
        first_events = schedule.groupby('tail_number').first().reset_index()
        first_events['destination_vertiport_id'] = first_events['origin_vertiport_id']
        return first_events
    
    @staticmethod
    def _get_initial_system_state_from_optimization(schedule, operation='flight_schedule') -> Tuple[pd.DataFrame, pd.DataFrame]:
        if operation == 'flight_schedule':
            # Get the rows that have aircraft_pushback_time = 0
            initial_states = schedule[schedule['aircraft_pushback_time'] == 0]
        elif operation == 'charging_schedule':
            # Get the rows that have charging_start_time = 0
            initial_states = schedule[schedule['charging_start_time'] == 0]
        else:
            raise ValueError(f"Operation {operation} is not defined correctly.")
        # Get the indices of the rows that have 0 start time
        initial_states_indices = initial_states.index

        return initial_states, initial_states_indices
    
    def _setup_scheduler_with_schedule(self, schedule):
        # Save the aircraft arrival schedule for the scheduler
        self.scheduler.flight_schedule = schedule
        self.scheduler.system_manager = self.system_manager

    @staticmethod
    def convert_schedule_columns_to_ms(schedule: pd.DataFrame, operation: str) -> pd.DataFrame:
        """
        Convert the columns of the aircraft arrival schedule to milliseconds.
        :param aircraft_schedule:
        :return:
        """
        # Convert the columns to milliseconds.
        if operation == 'flight_schedule':
            schedule['pushback_interarrival_time'] = sec_to_ms(schedule['pushback_interarrival_time'])
        elif operation == 'charging_schedule':
            schedule['charge_interarrival_time'] = sec_to_ms(schedule['charge_interarrival_time'])
        else:
            raise ValueError('Invalid operation type for reading schedule file.')
        return schedule
    
    # ---------------------------- Setup aircraft charging schedule ---------------------------- #

    def setup_charge_schedule(self) -> Union[pd.DataFrame, None]:
        if self.charge_schedule_file_path is not None:
            return self._load_charging_schedule_from_file()
        else:
            return None
        
    def _load_charging_schedule_from_file(self) -> pd.DataFrame:
        # Load and process aircraft arrival schedule
        schedule, initial_states = self._read_and_process_charging_schedule(self.charge_schedule_file_path)
        print("Success: Aircraft charging schedule loaded from file.")
        self._setup_scheduler_with_charge_schedule(schedule)
        return initial_states
    
    def _read_and_process_charging_schedule(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load aircraft arrival schedule from the path and sort by arrival time
        schedule = read_input_file(path).sort_values(by='charging_start_time')

        # Check if the columns of the input file are correct
        schedule = AircraftArrivalSetup.check_charging_schedule(schedule)

        # Convert the columns to milliseconds
        schedule = AircraftArrivalSetup.convert_schedule_columns_to_ms(schedule, operation='charging_schedule')

        # Get the initial state of the system from the input file
        initial_states, initial_states_indices = AircraftArrivalSetup._get_initial_system_state_from_optimization(
            schedule=schedule,
            operation='charging_schedule')
        schedule = schedule.drop(initial_states_indices)
        return schedule, initial_states
    
    @staticmethod
    def check_charging_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
        # Check if the columns of the input file are correct
        required_columns = [
            'vertiport_id',
            'init_soc',
            'target_soc',
            'charging_start_time',
            'process'
        ]
    
        check_none_columns = [
            'vertiport_id',
            'init_soc',
            'target_soc',
            'charging_start_time',
            'process'
        ]
    
        optional_columns = {'charge_interarrival_time': compute_interarrival_times_from_schedule(schedule['charging_start_time'])}
        return AircraftArrivalSetup.validate_columns(schedule, required_columns, optional_columns, check_none_columns)

    
    def _setup_scheduler_with_charge_schedule(self, schedule):
        # Save the aircraft charge schedule for the scheduler
        self.scheduler.charge_schedule = schedule
    
    # ---------------------------- Setup aircraft arrival schedule ---------------------------- #

    def create_aircraft_arrival(self):
        # Create aircraft arrival process.
        aircraft_schedule = self.setup_aircraft_schedule()
        self.logger.debug("Success: Aircraft arrival schedule loaded from file.")

        charge_schedule = self.setup_charge_schedule()
        self.logger.debug("Success: Aircraft charging schedule loaded from file.")

        if charge_schedule is None:
            self.logger.debug("No charging schedule is provided. The arrival schedule will read tail numbers from the file.")
            self.env.process(self._initial_aircraft_arrival_process_with_tail_number(aircraft_schedule))
            if self.flight_schedule_file_path is not None:
                self.logger.debug("Aircraft flight schedule is provided. The scheduler will initiate the schedule.")
                self.env.process(self.scheduler.initiate_fixed_schedule_from_file())             
        else:
            self.logger.debug("Flight schedule and a charging schedule is provided without tail numbers. The simulator will assign aircraft to the flights.")
            process_list = [
                self.env.process(self._initial_aircraft_arrival_process_without_tail_number(charge_schedule)),
                self.env.process(self.scheduler.initiate_fixed_schedule_from_file()),
                # self.env.process(self.scheduler.initiate_charge_schedule_from_file())
            ]
            self.env.all_of(process_list)
            self.scheduler.initiate_charge_schedule_from_file()



    def _initial_aircraft_arrival_process_with_tail_number(self, aircraft_schedule: pd.DataFrame):
        for tail_number in aircraft_schedule.tail_number:
            # Fields to retrieve for each aircraft.
            fields = ['aircraft_pushback_time', 'interarrival_time', 'soc', 'speed', 
                    'origin_vertiport_id', 'destination_vertiport_id', 'passengers_onboard', 'location',
                    'serviced_time_at_the_location', 'priority', 'process']

            # Create a dictionary to store the retrieved values.
            aircraft_data = {}

            for field in fields:
                # Get the value of the field for the aircraft.
                aircraft_data[field] = aircraft_schedule.loc[
                    aircraft_schedule.tail_number == tail_number, field].values[0]

            yield self.env.timeout(aircraft_data['interarrival_time'])

            self.env.process(
                # Simulate the aircraft.
                self.initiate_aircraft_arrival(tail_number=tail_number,
                                            soc=aircraft_data['soc'],
                                            input_cruise_speed=aircraft_data['speed'],
                                            origin_vertiport_id=aircraft_data['origin_vertiport_id'],
                                            destination_vertiport_id=aircraft_data['destination_vertiport_id'],
                                            passengers_onboard=aircraft_data['passengers_onboard'],
                                            location=aircraft_data['location'],
                                            serviced_time_at_the_location=aircraft_data['serviced_time_at_the_location'],
                                            departure_time=aircraft_data['aircraft_pushback_time'],
                                            priority=aircraft_data['priority'],
                                            initial_process=aircraft_data['process'],
                                            logger=self.logger,
                                            aircraft_logger=self.aircraft_logger)
            )       

    def _initial_aircraft_arrival_process_without_tail_number(self, charge_schedule: pd.DataFrame):
        for idx, row in charge_schedule.iterrows():
            yield self.env.timeout(row['charge_interarrival_time'])

            if lower_str(row['process']) == 'charging':
                initial_process = AircraftStatus.CHARGE
            elif lower_str(row['process']) == 'parking':
                initial_process = AircraftStatus.IDLE
            else:
                raise ValueError(f"Initial process {row['process']} is not supported. Input only CHARGING or PARKING.")
            
            self.logger.debug(f"Aircraft {idx} is arrived at the vertiport {row['vertiport_id']}." 
                             f" init_soc: {row['init_soc']}, target_soc: {row['target_soc']},"
                             f" process: {row['process']}")   

            self.env.process(
                # Simulate the aircraft.
                self.initiate_aircraft_arrival(departure_time=None,
                                               tail_number=idx,
                                               soc=row['init_soc'],
                                               target_soc=row['target_soc'],
                                               input_cruise_speed=0,
                                               origin_vertiport_id=row['vertiport_id'],
                                               destination_vertiport_id=row['vertiport_id'],
                                               passengers_onboard=[],
                                               location=row['location'],
                                               serviced_time_at_the_location=0,
                                               priority=1,
                                               initial_process=initial_process,
                                               logger=self.logger,
                                               aircraft_logger=self.aircraft_logger)
            )

    def initiate_aircraft_arrival(self,
                                  departure_time: int = None,
                                  tail_number: int = None,
                                  soc: int = 100,
                                  target_soc = None,
                                  input_cruise_speed: int = 72,
                                  origin_vertiport_id: str = None,
                                  destination_vertiport_id: str = None,
                                  passengers_onboard: List = None,
                                  location: str = None,
                                  serviced_time_at_the_location: int = 0,
                                  priority: int = 1,
                                  initial_process: str = AircraftStatus.IDLE,
                                  logger=None,
                                  aircraft_logger=None):
        
        if passengers_onboard is None:
            passengers_onboard = []

        aircraft = Aircraft(env=self.env,
                            tail_number=tail_number,
                            origin_vertiport_id=origin_vertiport_id,
                            destination_vertiport_id=destination_vertiport_id,
                            passengers_onboard=passengers_onboard,
                            location=location,
                            serviced_time_at_the_location=serviced_time_at_the_location,
                            departure_time=departure_time,
                            soc=soc,
                            target_soc=target_soc,
                            input_cruise_speed=input_cruise_speed,
                            arrival_time=self.env.now,
                            priority=priority,
                            initial_process=initial_process,
                            aircraft_params=self.aircraft_params,
                            system_manager=self.system_manager,
                            wind=self.wind,
                            event_saver=self.event_saver,
                            logger=logger,
                            aircraft_logger=aircraft_logger,
                            charging_strategy=self.charging_strategy
                            )
        
        self.event_saver.save_agent_state(agent=aircraft, agent_type='aircraft', event='agent_creation')
        # Save aircraft in aircraft_agent dictionary.
        self.system_manager.aircraft_agents[aircraft.tail_number] = aircraft

        self.vertiport_logger.info(f"Aircraft {aircraft.tail_number}, state: {aircraft.detailed_status}, SOC: {aircraft.soc}, target SOC: {aircraft.target_soc}, location: {aircraft.location}.")

        yield self.env.process(self.initiate_aircraft(aircraft=aircraft))
        
    def initiate_aircraft(self, aircraft: Aircraft):
        # Based on aircraft initial location, initiate the related process.    
        if aircraft.location in self.structural_entity_groups['fato']:
            aircraft.assigned_fato_id = aircraft.location
            yield self.env.process(self.system_manager.fato_and_parking_pad_usage_process(aircraft=aircraft))
        elif aircraft.location in self.structural_entity_groups['parking_pad']:
            aircraft.parking_space_id = aircraft.location
            yield self.env.process(self.system_manager.fato_and_parking_pad_usage_process(aircraft=aircraft))
        elif aircraft.location in self.structural_entity_groups['approach_fix_node']:
            yield self.env.process(self.system_manager.simulate_terminal_airspace_arrival_process(aircraft=aircraft,
                                                                                                     arriving_passengers=aircraft.passengers_onboard))
        elif aircraft.location in self.structural_entity_groups['departure_fix_node']:
            aircraft.flight_direction = f'{aircraft.origin_vertiport_id}_{aircraft.destination_vertiport_id}'
            yield self.env.process(self.system_manager.departure_fix_request_initial_state(aircraft=aircraft))
        else:
            raise ValueError(f'Unknown aircraft location: {aircraft.location}')