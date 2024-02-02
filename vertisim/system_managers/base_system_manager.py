import simpy
import networkx as nx
from typing import Any, Dict, Generator, Optional
from ..utils.units import sec_to_ms, ms_to_sec, ms_to_min, miles_to_m
from ..utils.distribution_generator import DistributionGenerator
from ..utils.weighted_random_chooser import random_choose_exclude_element
from ..utils.helpers import get_passenger_ids_from_passenger_list, miliseconds_to_hms, duplicate_str, \
    get_random_process_id, calculate_passenger_consolidation_time, check_whether_node_exists, careful_round
from ..aircraft.battery_model import BatteryModel
from ..aircraft.aircraft import Aircraft, AircraftStatus
import pandas as pd
from enum import Enum
from collections import defaultdict
from ..utils.calc_required_charge_time_from_required_energy import calc_required_charge_time_from_required_energy
from typing import List, Union

class BaseSystemManager:
    def __init__(self,
                 env: simpy.Environment,
                 vertiports: Dict,
                 vertiport_ids: List, 
                 vertiport_id_to_index_map,
                 num_initial_aircraft: int,                                 
                 scheduler: object,
                 wind: object,
                 airspace: object,
                 taxi_config: object,
                 sim_params: Dict,
                 output_params: Dict,
                 aircraft_params: Dict,
                 vertiport_distances: Dict,
                 passenger_distributions: Dict,                 
                 event_saver: object,
                 node_locations: Dict,
                 logger: object,
                 aircraft_logger: object,
                 passenger_logger: object,
                 sim_mode: str):
        self.env = env
        self.vertiports = vertiports
        self.vertiport_ids = vertiport_ids
        self.vertiport_id_to_index_map = vertiport_id_to_index_map
        self.num_initial_aircraft = num_initial_aircraft
        self.scheduler = scheduler
        self.wind = wind
        self.airspace = airspace
        self.taxi_config = taxi_config
        self.sim_params = sim_params
        self.output_params = output_params
        self.aircraft_params = aircraft_params
        self.vertiport_distances = vertiport_distances
        self.passenger_distributions = passenger_distributions
        self.event_saver = event_saver
        self.node_locations = node_locations
        self.logger = logger
        self.sim_mode = sim_mode
        self.aircraft_logger = aircraft_logger
        self.passenger_logger = passenger_logger
        self.aircraft_agents = {}
        self.passenger_agents = {}
        self.average_flight_durations_per_od_pair = self._initialize_avg_flight_durations()
        self.taxi_resource = simpy.Resource(self.env, 1)
        self.aircraft_battery_models = self.build_aircraft_battery_models()
        self.passenger_arrival_complete = False


    def get_aircraft_count(self) -> int:
        return sum(
            vertiport.num_aircraft
            for _, vertiport in self.vertiports.items()
        )    
    
    def get_available_aircraft_count(self, vertiport_id: str=None) -> int:
        if vertiport_id is not None:
            return len(self.vertiports[vertiport_id].available_aircraft_store.items)
        else:
            return sum(
                len(self.vertiports[vertiport_id].available_aircraft_store.items)
                for vertiport_id, _ in self.vertiports.items()
            )

    def build_aircraft_battery_models(self) -> Dict[Any, pd.DataFrame]:
        """
        Builds the battery charging lookup tables for the aircraft model for each vertiport.
        """
        # TODO: Currently, we support single aircraft model for all vertiports. However, each vertiport can
        # have different charger models. We might need to support different aircraft models in the future but not urgent/crucial
        return {vertiport_name: 
            BatteryModel(charger_model=vertiport.charger_model).
                        charge_process(battery_capacity=self.aircraft_params['battery_capacity'])
                for vertiport_name, vertiport in self.vertiports.items()
        }

    def _initialize_avg_flight_durations(self):
        # def inner_dict():
        #     return {'average_flight_duration': None, 'num_flights': 0, 'total_duration': 0}
    
        # # The outer defaultdict
        # dd = defaultdict(inner_dict)
        # for flight_direction in self.airspace.flight_directions:
        #     dd[flight_direction]
        # return dd
    
        def inner_dict(estimated_time):
            return {'average_flight_duration': estimated_time, 'num_flights': 0, 'total_duration': 0}
    
        # The outer defaultdict
        dd = defaultdict(lambda: inner_dict(None))
        for origin_id in self.vertiport_ids:
            for destination_id in self.vertiport_ids:
                if origin_id != destination_id:
                    estimated_time = self.initial_flight_duration_estimate(origin_id, destination_id, self.aircraft_params['cruise_speed'])
                    dd[f"{origin_id}_{destination_id}"] = inner_dict(estimated_time)
        return dd    
    
    def update_flight_duration(self, flight_direction: str, flight_duration: float):        
        # Update total duration and number of flights.
        self.increase_total_flight_duration(flight_direction, flight_duration)
        self.increase_flight_count(flight_direction)
        
        # Recalculate and update average flight duration.
        average_flight_duration = round(self.get_total_flight_time(flight_direction) / self.get_flight_count(flight_direction), 2)
        self.set_average_flight_time(flight_direction, average_flight_duration)

    def get_average_flight_time(self, flight_direction: str) -> float:
        """
        Retrieve the average flight time for the specified flight direction.
        """
        return self.average_flight_durations_per_od_pair[flight_direction]['average_flight_duration']

    def get_total_flight_time(self, flight_direction: str) -> float:
        """
        Retrieve the total flight time for the specified flight direction.
        """
        return self.average_flight_durations_per_od_pair[flight_direction]['total_duration']
    
    def get_flight_count(self, flight_direction: str) -> int:
        """
        Retrieve the number of flights for the specified flight direction.
        """
        return self.average_flight_durations_per_od_pair[flight_direction]['num_flights']
    
    def set_average_flight_time(self, flight_direction: str, average_time: float) -> None:
        """
        Set the average flight time for the specified flight direction.
        """
        self.average_flight_durations_per_od_pair[flight_direction]['average_flight_duration'] = average_time 

    def decrease_flight_count(self, flight_direction: str) -> None:
        """
        Decrease the number of flights for the specified flight direction.
        """
        self.average_flight_durations_per_od_pair[flight_direction]['num_flights'] -= 1
    
    def increase_flight_count(self, flight_direction: str) -> None:
        """
        Increase the number of flights for the specified flight direction.
        """
        self.average_flight_durations_per_od_pair[flight_direction]['num_flights'] += 1
    
    def increase_total_flight_duration(self, flight_direction: str, flight_duration: float) -> None:
        """
        Increase the total flight duration for the specified flight direction.
        """
        self.average_flight_durations_per_od_pair[flight_direction]['total_duration'] += flight_duration

    def initial_flight_duration_estimate(self, origin_id: str, destination_id: str, cruise_speed: float) -> float:
        """
        Compute the average flight time (in seconds) based on distance and aircraft speed.
        """
        flight_distance = miles_to_m(self.get_mission_length(origin_id, destination_id))
        return flight_distance / cruise_speed * 1.25        
    
    def assign_fato(self, vertiport_id: Any, parking_space: str, fato_id: str = None, rule: str = 'min') -> str:
        """
        Finds a FATO pad for the given parking space with the given rule. If there is no parking pad
        on the vertiport, choose the fato resource with zero users or the least queued fato

        Parameters
        ----------
        vertiport_id : Any
            Vertiport ID
        parking_space: str
            Assigned parking space for arriving aircraft
        rule: str
            Rule for assigning FATO pad. 'min' for searching closest fato pad

        Returns
        -------
        str
            assigned FATO ID for arriving aircraft
        """
        if fato_id is not None:
            return fato_id

        num_parking_pad = self.vertiports[vertiport_id].vertiport_layout.num_parking_pad
        if  num_parking_pad == 0:
            fato_pads = {}
            for fato in self.vertiports[vertiport_id].fato_store.items:
                # Choose the fato resource with the zero users or the least queued users
                fato_pads[fato.resource_id] = len(fato.fato_resource.queue)
                if fato.fato_resource.count == 0:
                    return fato.resource_id
            return min(fato_pads, key=fato_pads.get)
                    

        available_fato_pads = {}
        for fato in self.vertiports[vertiport_id].fato_store.items:
            # If that edge exists reserve the FATO
            if self.vertiports[vertiport_id].vertiport_layout.G.has_edge(fato.resource_id, parking_space):
                # FATO id's last digit needs to match with the parking pad's second to last digit
                # if (fato.resource_id[-1] == parking_space[-2])
                taxi_length = round(nx.dijkstra_path_length(self.vertiports[vertiport_id].vertiport_layout.G,
                                                      parking_space, fato.resource_id))
                available_fato_pads[fato.resource_id] = taxi_length

        # Find the key of shortest distance
        if rule == 'min':
            return min(available_fato_pads, key=available_fato_pads.get)
        else:
            return max(available_fato_pads, key=available_fato_pads.get)

    def request_fato(self, assigned_fato_resource: object, operation_type: str, priority: int = 1) -> object:
        """
        Requests the assigned fato pad resource from fato_store of vertiport manager

        Parameters
        ----------
        assigned_fato_resource : object
            Assigned TLOF pad resource by the vertiport manager.
        operation_type : str
            Operation type of the aircraft. 'arrival' or 'departure'

        Returns
        -------
        FatoResource class
            Class object that provides the FATO resource by request.
        """
        if operation_type == 'arrival':
            priority = self.sim_params['arrival_priority']
        elif operation_type == 'departure':
            priority = self.sim_params['departure_priority']
        else:
            raise ValueError("Operation type is not valid. Operation type should be 'arrival' or 'departure'.")
        return assigned_fato_resource.fato_resource.request(priority=priority)

    def call_fato_resource(self, vertiport_id: Any, assigned_fato_id: str) -> object:
        """
        Calls the assigned fato pad resource from fato_store of vertiport manager

        Parameters
        ----------
        vertiport_id : Any
            Vertiport ID
        assigned_fato_id : str
            Assigned TLOF pad ID by the vertiport manager.
        priority : int
            Priority for using the TLOF pad resource. Lower value prioritized

        Returns
        -------
        FatoResource class
            Class object that provides the FATO resource by request.
        """
        fato_list = [fato.resource_id for fato in self.vertiports[vertiport_id].fato_store.items]
        fato_index = fato_list.index(assigned_fato_id)
        return self.vertiports[vertiport_id].fato_store.items[fato_index]

    def get_taxi_instructions(self, vertiport_id: Any, assigned_fato_id: str, parking_space_id: str):
        """
        Gets the taxi route nodes and edges from vertiport manager
        
        Parameters
        ----------
        vertiport_id : Any
            Vertiport ID
        assigned_fato_id : str
            Assigned FATO ID by the vertiport manager.
        parking_space_id : str
            Assigned parking space ID by the vertiport manager.

        Returns
        -------
        taxi_route : list
            List of taxi route nodes
        taxi_route_edges : list
            List of taxi route edges
        """
        num_parking_pad = self.vertiports[vertiport_id].vertiport_layout.num_parking_pad
        if  num_parking_pad == 0:
            return [], []        
        # Get the taxi route nodes from vertiport manager
        taxi_route, _ = self.generate_taxi_route_nodes(vertiport_id, 
                                                       assigned_fato_id, 
                                                       parking_space_id)

        # Get the taxi route edges from vertiport manager
        taxi_route_edges = self.generate_taxi_route_edges(vertiport_id=vertiport_id, 
                                                          taxi_route=taxi_route)
        return taxi_route, taxi_route_edges

    def request_taxi_resource(self, vertiport_id: Any) -> Optional[simpy.resources.resource.Request]:
        """
        Requests the taxi resource from taxi manager

        Parameters
        ----------
        vertiport_id : Any
            Vertiport ID

        Returns
        -------
        taxi_usage_request : object
            Taxi usage request object

        """
        if self.taxi_config.is_simultaneous_taxi_and_take_off_allowed == False:
            return self.vertiports[vertiport_id].taxi_resource.request()
        else:
            return None

    def release_taxi_usage_request(self, vertiport_id, taxi_usage_request):
        """
        Releases the taxi resource from usage

        Parameters
        ----------
        taxi_usage_request : object
            Taxi usage request object

        Returns
        -------
        None
        """
        if taxi_usage_request is not None:
            self.vertiports[vertiport_id].taxi_resource.release(taxi_usage_request)

    def request_fix_resource(self, flight_direction: str, operation_type: str):
        """
        Requests the fix resource from vertiport manager
        """
        if operation_type == 'arrival':
            arrival_fix_resource = self.get_second_to_last_airlink_resource(flight_direction=flight_direction).airnode_resource
            return arrival_fix_resource.request(), arrival_fix_resource            
        elif operation_type == 'departure':
            departure_fix_resource = self.get_second_airlink_resource(flight_direction=flight_direction).airnode_resource
            return departure_fix_resource.request(), departure_fix_resource 
        
    def release_fix_resource(self, aircraft: Aircraft, operation: str) -> None:
        """
        Releases the fix resource from usage

        Parameters
        ----------
        aircraft : object
            Aircraft object
        operation : str
            Operation type of the aircraft. 'arrival' or 'departure'

        Returns
        -------
        None
        """
        if operation == 'arrival':
            aircraft.arrival_fix_resource.release(aircraft.arrival_fix_usage_request)
            aircraft.arriaval_fix_usage_request = None
            aircraft.arrival_fix_resource = None
        elif operation == 'departure':
            aircraft.departure_fix_resource.release(aircraft.departure_fix_usage_request)
            aircraft.departure_fix_usage_request = None
            aircraft.departure_fix_resource = None
        else:
            raise ValueError("Operation type is not valid. Operation type should be 'arrival' or 'departure'.")

    def get_first_airlink_resource(self, flight_direction: str):
        # First airlink resource is hover fix
        return next(iter(self.airspace.airlink_resources[flight_direction].values()))
    
    def get_first_airlink_node_id(self, flight_direction: str):
        # First airlink resource is hover fix
        return next(iter(self.airspace.airlink_resources[flight_direction].keys()))
    
    def get_second_airlink_resource(self, flight_direction: str):
        # Second airlink resource is the departure fix
        values_iterator = iter(self.airspace.airlink_resources[flight_direction].values())
        next(values_iterator)
        return next(values_iterator)

    def get_second_airlink_node_id(self, flight_direction: str):
        keys_iterator = iter(self.airspace.airlink_resources[flight_direction].keys())
        next(keys_iterator)
        return next(keys_iterator)    
    
    def get_third_airlink_resource(self, flight_direction: str):
        # Third airlink resource is the first waypoint
        values_iterator = iter(self.airspace.airlink_resources[flight_direction].values())
        next(values_iterator)
        next(values_iterator)
        return next(values_iterator)
    
    def get_third_airlink_node_id(self, flight_direction: str):
        keys_iterator = iter(self.airspace.airlink_resources[flight_direction].keys())
        next(keys_iterator)
        next(keys_iterator)
        return next(keys_iterator)

    def get_last_airlink_resource(self, flight_direction: str) -> simpy.Resource:
        # Last airlink resource is the hover descend fix
        return list(self.airspace.airlink_resources[flight_direction].values())[-1]
    
    def get_last_airlink_node_id(self, flight_direction: str) -> str:
        return list(self.airspace.airlink_resources[flight_direction].keys())[-1]
    
    def get_second_to_last_airlink_resource(self, flight_direction: str) -> simpy.Resource:
        # Second to last airlink resource is the final approach fix
        return list(self.airspace.airlink_resources[flight_direction].values())[-2]
    
    def get_second_to_last_airlink_node_id(self, flight_direction: str) -> str:
        return list(self.airspace.airlink_resources[flight_direction].keys())[-2]
    
    def get_third_to_last_airlink_resource(self, flight_direction: str) -> simpy.Resource:
        # Third to last airlink resource is the holding unit
        return list(self.airspace.airlink_resources[flight_direction].values())[-3]
    
    def get_third_to_last_airlink_node_id(self, flight_direction: str) -> str:
        return list(self.airspace.airlink_resources[flight_direction].keys())[-3]

    def generate_taxi_route_nodes(self, vertiport_id: str, source: str, target: str) -> Union[list, float]:
        """
        Returns a taxi route as a list between assigned FATO pad and parking
        space.

        Parameters
        ----------
        vertiport_id : str
            Vertiport ID
        source : str
            FATO pad ID or parking space ID assigned by the skyport manager
        target : str
            FATO pad ID or parking space ID assigned by the skyport manager

        Returns
        -------
        list
            list of nodes that an aircraft has to follow for its taxi
        float
            Total taxi length
        """
        num_parking_pad = self.vertiports[vertiport_id].vertiport_layout.num_parking_pad
        if  num_parking_pad == 0:
            return [], 0
        
        taxi_route = nx.astar_path(self.vertiports[vertiport_id].vertiport_layout.G, source, target)
        total_taxi_length = nx.astar_path_length(self.vertiports[vertiport_id].vertiport_layout.G, source, target)

        return taxi_route, round(total_taxi_length, 2)

    def generate_taxi_route_edges(self, vertiport_id, taxi_route: list) -> list:
        """
        Generates taxi route edges from taxi route nodes.

        Parameters
        ----------
        taxi_route : list
            list of nodes that an aircraft has to follow for its taxi

        Returns
        -------
        list
            list of edges that an aircraft has to follow for its taxi
        """
        num_parking_pad = self.vertiports[vertiport_id].vertiport_layout.num_parking_pad
        if  num_parking_pad == 0:
            return []
                
        return [
            {taxi_route[count], taxi_route[count + 1]}
            for count in range(len(taxi_route) - 1)
        ]

    def get_parking_pad(self, vertiport_id: Any, parking_space_id: Any = None) -> str:
        """
        Gets an available parking space from parking_space_store.

        Parameters
        ----------
        vertiport_id : str
            Vertiport ID

        Returns
        -------
        str
            Parking space ID
        """
        if parking_space_id is not None:
            if not check_whether_node_exists(self.node_locations, parking_space_id):
                raise ValueError(
                    f"Node {parking_space_id} does not exist in the vertiport {vertiport_id}"
                )
            return self.vertiports[vertiport_id].parking_space_store.get(lambda x: x.resource_id == parking_space_id)
        return self.vertiports[vertiport_id].parking_space_store.get()

    def release_parking_space(self, vertiport_id: Any, departure_parking_space: str) -> None:
        """
        Releases the parking space of the departing aircraft

        Parameters
        ----------
        vertiport_id : str
            Vertiport ID
        departure_parking_space : str
            Parking space ID of the departing aircraft

        Returns
        -------
        None
        """
        self.vertiports[vertiport_id].parking_space_store.put(departure_parking_space)

    def put_aircraft_into_available_aircraft_store(self, vertiport_id: Any, aircraft: object) -> None:
        """
        Puts the aircraft into the available aircraft store
        :param aircraft:
        :return:
        """
        self.vertiports[vertiport_id].available_aircraft_store.put(aircraft)

    def get_aircraft_from_available_aircraft_store(self, vertiport_id: Any, tail_number: str) -> object:
        """
        Gets the aircraft that has the lowest battery and can complete the mission
        :return: object
        """
        return self.vertiports[vertiport_id].available_aircraft_store.get(lambda aircraft: aircraft.tail_number == tail_number)

    def check_num_available_aircraft(self, vertiport_id: Any) -> int:
        """
        Checks the number of available aircraft in the available aircraft store
        :return: int
        """
        return len(self.vertiports[vertiport_id].available_aircraft_store.items)        
    
    def get_mission_length(self, origin_vertiport_id: Any, destination_vertiport_id: Any) -> float:
        """
        Returns the mission length given the destination vertiport ID
        """
        return self.vertiport_distances[(self.vertiport_distances['origin_vertiport_id'] == origin_vertiport_id) & 
                                        (self.vertiport_distances['destination_vertiport_id'] == destination_vertiport_id)]['distance_miles'].values[0]    

    def reserve_aircraft(self, 
                         origin_vertiport_id: Any, 
                         destination_vertiport_id: int, 
                         departing_passengers: list, 
                         tail_number: str = None,
                         soc: int = None):
        # Start tracking aircraft allocation time
        aircraft_allocation_start_time = self.env.now
        # Update the vertiport flight request state
        self.vertiports[origin_vertiport_id].num_flight_requests += 1           
        # Log the aircraft allocation process
        flight_id = get_random_process_id()
        self.logger.debug(f'|{flight_id}| Started: Aircraft allocation process at vertiport {origin_vertiport_id} for {destination_vertiport_id}. Aircraft tail number: {tail_number}, required SOC: {soc}')
        # Get the intended pushback time of the aircraft
        pushback_time = self.env.now             
        # Retrieve aircraft from the store
        aircraft = yield self.retrieve_aircraft(origin_vertiport_id, tail_number, soc) 
        # Set the status of the aircraft
        aircraft.status = AircraftStatus.FLY
        # Update the number of aircraft at the vertiport
        self.update_ground_aircraft_count(vertiport_id=aircraft.current_vertiport_id, update=-1)
        # Update the vertiport flight request state
        self.vertiports[origin_vertiport_id].num_flight_requests -= 1            
        # Set the real pushback time of the aircraft 
        aircraft.pushback_time = pushback_time
        # Set the flight direction of the aircraft
        aircraft.flight_direction = f'{origin_vertiport_id}_{destination_vertiport_id}'
        # Set the flight ID of the aircraft
        aircraft.flight_id = flight_id
        # Measure and save the aircraft allocation time
        self.save_aircraft_allocation_time(origin_vertiport_id, aircraft, aircraft_allocation_start_time)    

        # Check the waiting room for additional passengers if there is still space on the aircraft
        departing_passengers = self.add_additional_passengers_if_needed(origin_vertiport_id, destination_vertiport_id, departing_passengers)

        self.logger.debug(f'|{flight_id}| Completed: Aircraft allocation process at vertiport {origin_vertiport_id} for {destination_vertiport_id}. Aircraft tail number: {aircraft.tail_number}, SOC: {careful_round(aircraft.soc, 2)}, Departing passengers: {len(departing_passengers)}')

        # Save aircraft allocation and idle times
        self.save_aircraft_times(origin_vertiport_id, aircraft)

        # Assign passengers to the aircraft
        aircraft.passengers_onboard = departing_passengers

        # Save total passenger waiting time and flight assignment time
        self.save_passenger_times(origin_vertiport_id=origin_vertiport_id, passengers=aircraft.passengers_onboard) 
        self.passenger_logger.info(f"Passenger waiting times for flight from {origin_vertiport_id} to {destination_vertiport_id}: Passenger ids : {[p.passenger_id for p in departing_passengers]} waiting times: {[miliseconds_to_hms(self.env.now - passenger.waiting_room_arrival_time) for passenger in departing_passengers]}")

        # Start the passenger departure process
        if departing_passengers:
            self.logger.debug(f'|{flight_id}| Started: Passengers exited the waiting room at vertiport {origin_vertiport_id} for {destination_vertiport_id} flight.')
            self.env.process(self.initiate_passenger_departure(departing_passengers=departing_passengers))
            self.logger.debug(f'|{flight_id}| Completed: Passengers arrived at the boarding gate at vertiport {origin_vertiport_id} for {destination_vertiport_id} flight.')

        # Start the aircraft departure process
        yield self.env.process(
            self.simulate_aircraft_departure_process(aircraft=aircraft,
                                                    origin_vertiport_id=origin_vertiport_id, 
                                                    destination_vertiport_id=destination_vertiport_id)
        )    

    def add_additional_passengers_if_needed(self, origin_vertiport_id, destination_vertiport_id, departing_passengers):
        waiting_room = self.vertiports[origin_vertiport_id].waiting_room_stores[destination_vertiport_id]
        self.passenger_logger.info(f"Checking for additional passengers to fill the aircraft at {origin_vertiport_id}."
                                    f" Passengers in the waiting room: {[p.passenger_id for p in waiting_room.items]}"
                                    f" Their waiting room arrival times: {[miliseconds_to_hms(p.waiting_room_arrival_time) for p in waiting_room.items]}"
                                    f" Their waiting times (min): {[ms_to_min(self.env.now - passenger.waiting_room_arrival_time) for passenger in waiting_room.items]}") 
               
        if len(departing_passengers) < self.aircraft_params['pax'] and len(waiting_room.items) > 0:
            departing_passengers.extend(self.scheduler.last_call_check(
                current_waiting_room=waiting_room,
                num_departing_passengers=len(departing_passengers))) 
        self.passenger_logger.info(f"Num additional passengers can be allocated: {self.aircraft_params['pax'] - len(departing_passengers)}")  
        return departing_passengers        

    def retrieve_aircraft(self, origin_vertiport_id, tail_number=None) -> object:
        """
        Retrieves an aircraft from the available aircraft store
        :return: object

        Override this method in the subclass if needed.
        """
        if tail_number is not None:
            return self.vertiports[origin_vertiport_id].available_aircraft_store.get(lambda aircraft: aircraft.tail_number == tail_number)
        return self.vertiports[origin_vertiport_id].available_aircraft_store.get()
    
    def save_aircraft_allocation_time(self, origin_vertiport_id, aircraft, start_time):
        aircraft.flight_allocation_time = self.env.now
        self.event_saver.save_aircraft_allocation_time(vertiport_id=origin_vertiport_id,
                                                    tail_number=aircraft.tail_number,
                                                    allocation_time=aircraft.flight_allocation_time - start_time)

    def save_aircraft_times(self, origin_vertiport_id: str, aircraft: Aircraft) -> None:
        self.event_saver.save_agent_state(agent=aircraft, agent_type='aircraft', event='aircraft_allocation')
        if aircraft.charging_end_time is None:
            # At the beginning of the simulation, charging_end_time is None
            if not aircraft.charged_during_turnaround:
                idle_time = self.env.now - aircraft.arrival_time
                self.event_saver.save_aircraft_idle_time(vertiport_id=origin_vertiport_id,
                                                        idle_time=idle_time)
                aircraft.save_process_time(event='idle', process_time=idle_time)
                self.logger.info(f'Saved idle time of {miliseconds_to_hms(idle_time)} between initial arrival and first flight for aircraft {aircraft.tail_number} at {origin_vertiport_id}.')
        else:
            if not aircraft.charged_during_turnaround:
                idle_time = aircraft.env.now - aircraft.arrival_time
            else:
                idle_time = aircraft.env.now - aircraft.charging_end_time
            aircraft.idle_time += idle_time
            self.event_saver.save_aircraft_idle_time(vertiport_id=origin_vertiport_id,
                                                     idle_time=aircraft.idle_time)
            aircraft.save_process_time(event='idle', process_time=idle_time)
            self.logger.info(f'Saved idle time {miliseconds_to_hms(idle_time)} between charging end or arrival and the flight allocation for aircraft {aircraft.tail_number} at {origin_vertiport_id}.')            

    def parking_pad_request(self, aircraft: object, parking_space_id: Any = None):
        # parking space reservation for landing
        parking_space = yield self.get_parking_pad(vertiport_id=aircraft.destination_vertiport_id, parking_space_id=parking_space_id)
        parking_space_id = parking_space.resource_id
        # Save them to use for departure
        aircraft.parking_space = parking_space
        aircraft.parking_space_id = parking_space_id

    def fato_pad_request(self, aircraft: object, operation_type: str, fato_id: str = None):
        if operation_type == 'arrival':
            vertiport_id = aircraft.destination_vertiport_id
        elif operation_type == 'departure':
            vertiport_id = aircraft.origin_vertiport_id
        else:
            raise ValueError('Operation type for FATO request is not valid.')
        # Currently, vertiport manager assigns the closest FATO
        assigned_fato_id = self.assign_fato(vertiport_id=vertiport_id,
                                            parking_space=aircraft.parking_space_id,
                                            fato_id=fato_id)
        aircraft.assigned_fato_id = assigned_fato_id

        # Aircraft calls the FATO resource that will be requested.
        assigned_fato_resource = self.call_fato_resource(vertiport_id=vertiport_id,
                                                        assigned_fato_id=assigned_fato_id)
        aircraft.assigned_fato_resource = assigned_fato_resource

        if operation_type == 'departure':
            # Increase the aircraft departure queue counter
            self.event_saver.update_aircraft_departure_queue_counter(vertiport_id=vertiport_id,
                                                                    queue_update=1)   
            self.vertiports[vertiport_id].aircraft_departure_queue_count += 1

        # Request for FATO pad usage
        aircraft_fato_usage_request = self.request_fato(
            assigned_fato_resource=assigned_fato_resource,
            operation_type=operation_type)
        yield aircraft_fato_usage_request
        aircraft.aircraft_fato_usage_request = aircraft_fato_usage_request

        if operation_type == 'departure':
            # Increase the aircraft departure queue counter
            self.event_saver.update_aircraft_departure_queue_counter(vertiport_id=vertiport_id,
                                                                    queue_update=-1)    
            self.vertiports[vertiport_id].aircraft_departure_queue_count -= 1 

        if operation_type == 'arrival' and \
            self.vertiports[aircraft.destination_vertiport_id].vertiport_layout.num_parking_pad == 0:
            # If there is no parking pad then aircraft will be parked on FATO
            aircraft.parking_space = assigned_fato_resource
            aircraft.parking_space_id = assigned_fato_id  
            parking_space = assigned_fato_resource
            parking_space_id = assigned_fato_id

    def simulate_landing_process(self, aircraft: object):
        # Save FATO usage start time
        self.event_saver.update_fato_usage_tracker(vertiport_id=aircraft.destination_vertiport_id,
                                                   fato_usage=1)
        
        # Landing process
        yield self.env.process(aircraft.aircraft_landing_process())

        # Release approach fix
        approach_fix = self.get_last_airlink_resource(flight_direction=aircraft.flight_direction).airnode_resource
        approach_fix.release(aircraft.arrival_fix_usage_request)

    def simulate_taxi_process(self, aircraft: object, operation_type: str):
        # Get the taxi route
        if operation_type == 'arrival':
            taxi_route, taxi_route_edges = self.get_taxi_instructions(
                vertiport_id=aircraft.destination_vertiport_id,
                assigned_fato_id=aircraft.assigned_fato_id,
                parking_space_id=aircraft.parking_space_id
            )
        elif operation_type == 'departure':
            taxi_route, taxi_route_edges = self.get_taxi_instructions(
                vertiport_id=aircraft.origin_vertiport_id,
                assigned_fato_id=aircraft.assigned_fato_id,
                parking_space_id=aircraft.parking_space_id
            )

        taxi_start_time = self.env.now

        # Taxi process
        yield self.env.process(
            aircraft.taxi(
                taxi_route=taxi_route,
                taxi_route_edges=taxi_route_edges,
                operation_type=operation_type)
        )
        taxi_end_time = self.env.now
        self.save_taxi_time(aircraft=aircraft, taxi_time=taxi_end_time-taxi_start_time)
        aircraft.save_process_time(event='taxi', process_time=taxi_end_time-taxi_start_time)

    def update_flight_info(self, aircraft: object):
        self.event_saver.update_flight_tracker(vertiport_id=aircraft.origin_vertiport_id)
        self.event_saver.update_load_factor_tracker(vertiport_id=aircraft.origin_vertiport_id, 
                                                    load_factor=len(aircraft.passengers_onboard))
        
    def save_arrival_departure_time(self, aircraft: object, operation_type: str):
        if operation_type == 'arrival':
            aircraft.arrival_time = self.env.now
            aircraft.flight_duration = aircraft.arrival_time - aircraft.departure_time
        elif operation_type == 'departure':
            aircraft.departure_time = self.env.now   
    
    def departure_fix_request(self, aircraft):
        # Request departure fix resource
        departure_fix_usage_request, departure_fix_resource = self.request_fix_resource(flight_direction=aircraft.flight_direction,
                                                                operation_type='departure')
        yield departure_fix_usage_request
        aircraft.departure_fix_usage_request = departure_fix_usage_request
        aircraft.departure_fix_resource = departure_fix_resource

    def departure_fix_request_initial_state(self, aircraft):
        # Request departure fix resource
        departure_fix_usage_request, departure_fix_resource = self.request_fix_resource(flight_direction=aircraft.flight_direction, 
                                                                operation_type='departure')
        yield departure_fix_usage_request
        aircraft.departure_fix_usage_request = departure_fix_usage_request
        aircraft.departure_fix_resource = departure_fix_resource

        yield self.env.process(aircraft.fly_in_the_airspace())

    def simulate_aircraft_departure_process(self, aircraft, origin_vertiport_id, destination_vertiport_id):
        aircraft.origin_vertiport_id = origin_vertiport_id
        aircraft.destination_vertiport_id = destination_vertiport_id

        # Save passenger boarding time
        for passenger in aircraft.passengers_onboard:
            self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='boarding_start')

        if not aircraft.charged_during_turnaround:
            yield self.env.process(aircraft.embark_passengers())
            aircraft.save_process_time(event='boarding', process_time=self.env.now-aircraft.flight_allocation_time)

        # Save aircraft boarding time
        boarding_end_time = self.env.now

        # Save passenger departure queue waiting time
        for passenger in aircraft.passengers_onboard:
            self.event_saver.save_passenger_departure_queue_waiting_time(
                vertiport_id=aircraft.origin_vertiport_id,
                passenger_id=passenger.passenger_id,
                boarding_time=boarding_end_time)

        yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='departure'))

        if not self.sim_params['only_aircraft_simulation'] and len(aircraft.passengers_onboard) > 0:
            # Decrease the passenger departure queue counter
            self.event_saver.update_passenger_departure_queue_counter(vertiport_id=aircraft.origin_vertiport_id,
                                                                      queue_update=-1)

        # Save FATO usage start time
        self.event_saver.update_fato_usage_tracker(vertiport_id=aircraft.origin_vertiport_id,
                                                   fato_usage=1)

        taxi_usage_request = self.request_taxi_resource(vertiport_id=aircraft.origin_vertiport_id)
        if taxi_usage_request is not None:
            yield taxi_usage_request

        # Save pushback time
        self.event_saver.save_agent_state(agent=aircraft, agent_type='aircraft', event='pushback') 
        self.event_saver.save_flight_info(origin_vertiport_id=aircraft.origin_vertiport_id,
                                          destination_vertiport_id=aircraft.destination_vertiport_id,
                                          tail_number=aircraft.tail_number,
                                          flight_id=aircraft.flight_id,
                                          aircraft_pushback_time=self.env.now,
                                          passengers=get_passenger_ids_from_passenger_list(aircraft.passengers_onboard),
                                          aircraft_model=aircraft.aircraft_model,
                                          event_type='pushback')   

        # Save total transfer time of a passenger
        for passenger in aircraft.passengers_onboard:
            self.event_saver.save_passenger_transfer_time(vertiport_id=aircraft.origin_vertiport_id,
                                                          passenger_id=passenger.passenger_id,
                                                          transfer_time=self.env.now - passenger.vertiport_arrival_time)

        # Taxi-out process
        yield self.env.process(self.simulate_taxi_process(aircraft=aircraft, operation_type='departure'))

        
        yield self.env.process(self.departure_fix_request(aircraft=aircraft))

        self.release_taxi_usage_request(vertiport_id=aircraft.origin_vertiport_id,
                                        taxi_usage_request=taxi_usage_request)   

        self.update_flying_aircraft_count(update=1)
        
        # Take-off process
        yield self.env.process(
            aircraft.aircraft_take_off_process(
                take_off_fato_id=aircraft.assigned_fato_id
            )
        ) 

        self.update_flight_info(aircraft=aircraft)

        yield self.env.process(aircraft.fly_in_the_airspace())

    def update_ground_aircraft_count(self, vertiport_id: str, update: int):
        self.update_ground_aircraft_count_at_the_vertiport(vertiport_id=vertiport_id, update=update)
        self.update_event_saver_ground_aircraft_counter()

    def update_ground_aircraft_count_at_the_vertiport(self, vertiport_id: str, update: int):
        self.vertiports[vertiport_id].num_aircraft += update

    def update_event_saver_ground_aircraft_counter(self):
        aircraft_counts = {
            vertiport_id: vertiport.num_aircraft
            for vertiport_id, vertiport in self.vertiports.items()
        }
        self.event_saver.update_ground_aircraft_counter(aircraft_counts)

    def update_flying_aircraft_count(self, update: int):
        self.event_saver.update_flying_aircraft_counter(update)

    def save_passenger_trip_times(self, aircraft: object, flight_direction: str):
        for passenger in aircraft.passengers_onboard:
            self.event_saver.save_passenger_trip_time(agent_id=passenger.passenger_id,
                                                      flight_direction=flight_direction,
                                                      trip_time=self.env.now - passenger.vertiport_arrival_time)
            
    def save_taxi_time(self, aircraft, taxi_time):
        self.event_saver.save_aircraft_taxi_time(vertiport_id=aircraft.destination_vertiport_id,
                                                 taxi_time=taxi_time)

    def save_passenger_times(self, origin_vertiport_id, passengers):
        for passenger in passengers:
            self.event_saver.save_passenger_waiting_time(vertiport_id=origin_vertiport_id,
                                                        passenger_id=passenger.passenger_id,
                                                        waiting_time=self.env.now - passenger.waiting_room_arrival_time)
            self.save_flight_assignment_times(passenger)

    def save_flight_assignment_times(self, passenger):
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='flight_assignment')
        passenger.flight_assignment_time = self.env.now

    def initiate_passenger_departure(self, departing_passengers):
        yield self.env.process(self.simulate_passenger_departure(departing_passengers=departing_passengers))    

    def get_available_aircraft_tail_numbers(self, vertiport_id: str) -> list:
        return [aircraft.tail_number for aircraft in self.vertiports[vertiport_id].available_aircraft_store.items]
    
    def is_all_aircraft_idle(self):
        return all(
            aircraft.status == AircraftStatus.IDLE
            for aircraft in self.aircraft_agents.values()
        )
    

    # PASSENGER RELATED METHODS
    # -------------------------
    def simulate_passenger_arrival(self, passenger):
        """ Simulate passenger."""
        # Save passenger creation time
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='vertiport_entrance')
        passenger.vertiport_arrival_time = self.env.now

        # Put the passenger into the waiting room
        yield self.env.process(self.put_passenger_into_waiting_room(passenger))

    def simulate_passenger_departure(self, departing_passengers):
        yield self.env.process(departing_passengers[0].waiting_room_to_boarding_gate())
        for passenger in departing_passengers:
            passenger.location = f'{passenger.origin_vertiport_id}_GATE'
            self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='boarding_gate')  

    def get_total_waiting_passenger_count(self):
        return sum(vertiport.get_passenger_count() for vertiport in self.vertiports.values())
    
    def check_all_waiting_rooms_empty(self):
        return all(vertiport.is_waiting_room_empty() for vertiport in self.vertiports.values())