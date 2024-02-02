from .base_system_manager import BaseSystemManager
import simpy
import networkx as nx
from typing import Any, Dict, Generator, Optional
from ..utils.units import sec_to_ms, ms_to_sec, ms_to_min, miles_to_m
from ..utils.distribution_generator import DistributionGenerator
from ..utils.weighted_random_chooser import random_choose_exclude_element
from ..utils.helpers import get_passenger_ids_from_passenger_list, miliseconds_to_hms, duplicate_str, \
    get_random_process_id, calculate_passenger_consolidation_time, check_whether_node_exists, careful_round, flatten_dict, write_to_db
from ..aircraft.battery_model import BatteryModel
from ..aircraft.aircraft import Aircraft, AircraftStatus
from ..utils.get_state_variables import get_simulator_states
import pandas as pd
from enum import Enum
from collections import defaultdict
from ..utils.calc_required_charge_time_from_required_energy import calc_required_charge_time_from_required_energy
from typing import List, Union

class OnDemandSystemManager(BaseSystemManager):
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
                 sim_mode: str = 'ondemand'):

        super().__init__(env=env,
                        vertiports=vertiports,
                        vertiport_ids=vertiport_ids,
                        vertiport_id_to_index_map=vertiport_id_to_index_map,
                        num_initial_aircraft=num_initial_aircraft,
                        scheduler=scheduler,
                        wind=wind,
                        airspace=airspace,
                        taxi_config=taxi_config,
                        sim_params=sim_params,
                        output_params=output_params,
                        aircraft_params=aircraft_params,
                        vertiport_distances=vertiport_distances,
                        passenger_distributions=passenger_distributions,
                        event_saver=event_saver,
                        node_locations=node_locations,
                        logger=logger,
                        aircraft_logger=aircraft_logger,
                        passenger_logger=passenger_logger,
                        sim_mode=sim_mode
                        )
        self.aircraft_agents = {}
        self.passenger_agents = {}
        self.taxi_resource = simpy.Resource(self.env, 1)
        self.aircraft_battery_models = self.build_aircraft_battery_models()   
        self.charging_time_distribution = self.build_charging_time_distribution()   
        if (
            not self.sim_params['only_aircraft_simulation']
            and self.sim_params['max_passenger_waiting_time']
        ):
            self.env.process(self.check_passenger_max_waiting_time_threshold())


    def build_charging_time_distribution(self) -> Optional[DistributionGenerator]:
        """
        Builds the charging time distribution with the given parameters
        """
        if self.aircraft_params['charging_time_dist'] is None:
            return None
        return DistributionGenerator(self.aircraft_params['charging_time_dist'])         
    
    def trigger_scheduler(self, 
                          origin_vertiport_id: str, 
                          destination_vertiport_id: int, 
                          passengers_waiting_room: simpy.Store) -> Generator:
        if departing_passengers := self.scheduler.check_waiting_room(current_waiting_room=passengers_waiting_room):
            # Save passenger group completion time
            for passenger in departing_passengers:
                self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='passenger_consolidation')

            # Save passenger consolidation time
            self.event_saver.save_passenger_group_consolidation_time(
                vertiport_id=origin_vertiport_id,
                passenger_ids=get_passenger_ids_from_passenger_list(departing_passengers),
                wr_number=destination_vertiport_id,
                consolidation_time=calculate_passenger_consolidation_time(departing_passengers)
            )

            self.logger.debug(f"Passengers {[passenger.passenger_id for passenger in departing_passengers]} are ready to depart from {origin_vertiport_id} to {destination_vertiport_id}")

            # Put passengers into the flight request queue
            for passenger in departing_passengers:
                self.vertiports[origin_vertiport_id].flight_request_stores[destination_vertiport_id].put(passenger)
                passenger.flight_queue_store = self.vertiports[origin_vertiport_id].flight_request_stores[destination_vertiport_id]

            # Increase passenger queue count
            self.event_saver.update_passenger_departure_queue_counter(vertiport_id=origin_vertiport_id, queue_update=1)

            yield self.env.process(self.reserve_aircraft(
                origin_vertiport_id=origin_vertiport_id,
                destination_vertiport_id=destination_vertiport_id,
                departing_passengers=departing_passengers)
                )
        
        if self.sim_params['fleet_rebalancing']:
            num_available_aircraft_at_origin = self.check_num_available_aircraft(origin_vertiport_id)
            num_available_aircraft_at_dest = self.check_num_available_aircraft(destination_vertiport_id)
            num_pax_groups = int(self.get_num_waiting_passengers(origin_vertiport_id) / self.aircraft_params['pax'])
            required_aircraft = max(num_pax_groups - num_available_aircraft_at_origin, 0)
            self.logger.debug(f"# of available aircraft at {origin_vertiport_id}: {num_available_aircraft_at_origin}/{self.vertiports[origin_vertiport_id].num_aircraft}, # of available aircraft at {destination_vertiport_id}: {num_available_aircraft_at_dest}/{self.vertiports[destination_vertiport_id].num_aircraft}")
            self.logger.debug(f"Required aircraft for rebalancing: {required_aircraft}. # of pax groups: {num_pax_groups}")
            if required_aircraft > 0 and num_available_aircraft_at_dest > 0:
                self.event_saver.update_repositioning_counter(vertiport_id=destination_vertiport_id,
                                                repositioning_count=required_aircraft)
                self.logger.debug(f"Creating {required_aircraft} empty flights to fulfill the demand at {origin_vertiport_id}")
                for _ in range(required_aircraft):
                    self.env.process(self.reserve_aircraft(origin_vertiport_id=destination_vertiport_id,
                                                            destination_vertiport_id=origin_vertiport_id,
                                                            departing_passengers=[]))                

    def get_num_waiting_passengers(self, origin_vertiport_id: str):
        """
        Checks the demand for the given vertiport
        """
        return self.vertiports[origin_vertiport_id].get_passenger_count()

    def reserve_aircraft(self, 
                         origin_vertiport_id: Any, 
                         destination_vertiport_id: int, 
                         departing_passengers: list):
        # Start tracking aircraft allocation time
        aircraft_allocation_start_time = self.env.now
        # Update the vertiport flight request state
        self.vertiports[origin_vertiport_id].num_flight_requests += 1           
        # Log the aircraft allocation process
        flight_id = get_random_process_id()
        self.logger.debug(f'|{flight_id}| Started: Aircraft allocation process at vertiport {origin_vertiport_id} for {destination_vertiport_id}.')
        # Get the intended pushback time of the aircraft
        pushback_time = self.env.now             
        # Retrieve aircraft from the store
        # yield self.env.timeout(1)
        aircraft = yield self.retrieve_aircraft(origin_vertiport_id) 
        # Set the status of the aircraft
        aircraft.status = AircraftStatus.FLY
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

        # Remove passengers from the flight queue
        self.scheduler.pop_passengers_from_flight_queue_by_id(passengers=departing_passengers)

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

    
    # def add_additional_passengers_if_needed(self, origin_vertiport_id, destination_vertiport_id, departing_passengers):
    #     waiting_room = self.vertiports[origin_vertiport_id].waiting_room_stores[destination_vertiport_id]
    #     self.passenger_logger.info(f"Checking for additional passengers to fill the aircraft at {origin_vertiport_id}."
    #                                 f" Passengers in the waiting room: {[p.passenger_id for p in waiting_room.items]}"
    #                                 f" Their waiting room arrival times: {[miliseconds_to_hms(p.waiting_room_arrival_time) for p in waiting_room.items]}"
    #                                 f" Their waiting times (min): {[ms_to_min(self.env.now - passenger.waiting_room_arrival_time) for passenger in waiting_room.items]}") 
               
    #     if len(departing_passengers) < self.aircraft_params['pax'] and len(waiting_room.items) > 0:
    #         departing_passengers.extend(self.scheduler.last_call_check(
    #             current_waiting_room=waiting_room,
    #             num_departing_passengers=len(departing_passengers))) 
    #     self.passenger_logger.info(f"Num additional passengers can be allocated: {self.aircraft_params['pax'] - len(departing_passengers)}")  
    #     return departing_passengers
    
    def check_passenger_max_waiting_time_threshold(self):
        """
        Checks all of the waiting rooms. If there are passengers who are waiting
        more than self.sim_params['max_passenger_waiting_time'], it creates a flight for those passengers immediately.
        """
        while True:
            # Check the waiting rooms every max_waiting_time/2.
            yield self.env.timeout(sec_to_ms(self.sim_params['max_passenger_waiting_time']) / 2)

            if departing_passenger_groups := self.scheduler.check_max_waiting_time_threshold(
                    sec_to_ms(self.sim_params['max_passenger_waiting_time'])):

                for origin in departing_passenger_groups:
                    for destination in departing_passenger_groups[origin]:
                        departing_passengers = departing_passenger_groups[origin][destination]
                        for passenger in departing_passengers:
                            self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='passenger_waiting_room_departure') 
                            # Put passengers into the flight request queue
                            self.vertiports[origin].flight_request_stores[destination].put(passenger)
                            passenger.flight_queue_store = self.vertiports[origin].flight_request_stores[destination]  
                        # Increase passenger queue count
                        self.event_saver.update_passenger_departure_queue_counter(vertiport_id=origin, 
                                                                                queue_update=len(departing_passenger_groups))   
                        flight_queue = [self.env.process(
                            self.reserve_aircraft(origin_vertiport_id=origin, 
                                                destination_vertiport_id=departing_passengers[0].destination_vertiport_id,
                                                departing_passengers=departing_passengers))
                        ]

                    yield self.env.all_of(flight_queue) 

    def put_passenger_into_waiting_room(self, passenger: object) -> None:
        """
        Puts the passenger into the waiting room based on their flight destination and triggers scheduler
        :param passenger:
        :return:
        """
        # Define the location of the passenger as the waiting room at their current location
        passenger.location = f'{passenger.origin_vertiport_id}_ROOM'
        # Save the current state of the passenger
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='enter_waiting_room')
        # Record the current time as the passenger's waiting room arrival time
        passenger.waiting_room_arrival_time = self.env.now
        # Get the waiting room that corresponds to the passenger's destination
        passengers_waiting_room = self.vertiports[passenger.origin_vertiport_id].waiting_room_stores[passenger.destination_vertiport_id]
        # Put the passenger into the waiting room and keep a reference to it
        passengers_waiting_room.put(passenger)
        passenger.waiting_room_store = passengers_waiting_room
        # Log the passenger's arrival
        self.passenger_logger.info(f'Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Currently waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}.')

        if self.sim_params['training_data_collection']:
            self.save_state_variables_for_training(passenger_id=passenger.passenger_id)

        # print(f"Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}. Flight queue: {[p.passenger_id for p in self.vertiports[passenger.origin_vertiport_id].flight_request_stores[passenger.destination_vertiport_id].items]}")
        # Trigger scheduler
        yield self.env.process(self.trigger_scheduler(origin_vertiport_id=passenger.origin_vertiport_id,
                                                    destination_vertiport_id=passenger.destination_vertiport_id, 
                                                    passengers_waiting_room=passengers_waiting_room))  

    def save_state_variables_for_training(self, passenger_id):
        states = get_simulator_states(vertiports=self.vertiports,
                                      aircraft_agents=self.aircraft_agents,
                                      num_initial_aircraft=self.num_initial_aircraft,
                                      simulation_states=self.sim_params['simulation_states'])
        states['passenger_id'] = passenger_id
        states['sim_id'] = self.sim_params['sim_id']
        states = flatten_dict(states)
        write_to_db(db_path=self.output_params['state_trajectory_db_path'], 
                    table_name=self.output_params['state_trajectory_db_tablename'],
                    dic=states)
        
    def check_holding_aircraft(self, flight_direction: str, origin_vertiport_id: Any, destination_vertiport_id: Any):
        arrival_fix_resource = self.get_second_to_last_airlink_resource(flight_direction=flight_direction).airnode_resource
        if len(arrival_fix_resource.queue) > 0:
            num_queued = len(arrival_fix_resource.queue)
            num_available_aircraft_at_dest = self.check_num_available_aircraft(destination_vertiport_id)
            num_available_aircraft_at_origin = self.check_num_available_aircraft(origin_vertiport_id)
            # additional_repositioning = 0
            # if (num_available_aircraft_at_origin+1) * 2 < num_available_aircraft_at_dest:
            #     additional_repositioning += 1
            num_scheduled_for_repositioning = min(num_queued, num_available_aircraft_at_dest) 
            self.event_saver.update_repositioning_counter(vertiport_id=destination_vertiport_id,
                                                          repositioning_count=num_scheduled_for_repositioning)
            # num_scheduled_for_repositioning += additional_repositioning
            self.logger.debug(f'Number of holding aircraft queued at {destination_vertiport_id}: {num_queued}. Number of available aircraft at {destination_vertiport_id}: {num_available_aircraft_at_dest}')
            for _ in range(num_scheduled_for_repositioning):
                self.env.process(self.assign_empty_flight(vertiport_id=destination_vertiport_id))

    def assign_empty_flight(self, vertiport_id: Any):
        # TODO: Check the waiting rooms and the parking pad availability of the other vertiports and 
        # pick the one with the highest number of passengers
        destination_id = random_choose_exclude_element(elements_list=self.vertiport_ids, 
                                                       exclude_element=vertiport_id, 
                                                       num_selection=1)[0]
        num_available_aircraft_at_origin = self.check_num_available_aircraft(vertiport_id)
        num_available_aircraft_at_destination = self.check_num_available_aircraft(destination_id)

        self.logger.debug(f'Assigning empty flight from {vertiport_id} to {destination_id}.'
                         f' Number of available aircraft at {vertiport_id}: {num_available_aircraft_at_origin}. Number of available aircraft at {destination_id}: {num_available_aircraft_at_destination}.')
        yield self.env.process(self.reserve_aircraft(origin_vertiport_id=vertiport_id,
                                                destination_vertiport_id=destination_id,
                                                departing_passengers=[]))        
        

    def simulate_terminal_airspace_arrival_process(self, aircraft: object, arriving_passengers: list):
        holding_start = self.env.now
        
        # Increase aircraft arrival queue counter
        self.event_saver.update_aircraft_arrival_queue_counter(vertiport_id=aircraft.destination_vertiport_id,
                                                               queue_update=1)
        
        # Request arrival fix resource
        arrival_fix_usage_request, arrival_fix_resource = self.request_fix_resource(flight_direction=aircraft.flight_direction, operation_type='arrival')
        self.logger.debug(f'Aircraft {aircraft.tail_number} requesting arrival fix resource at {aircraft.destination_vertiport_id}.'
                         f' Number of holding aircraft queued at {aircraft.destination_vertiport_id}: {len(arrival_fix_resource.queue)}'
                         f' Number of available aircraft at {aircraft.destination_vertiport_id}: {self.check_num_available_aircraft(aircraft.destination_vertiport_id)}')
        if self.sim_params['fleet_rebalancing']:
            self.check_holding_aircraft(flight_direction=aircraft.flight_direction,
                                        origin_vertiport_id=aircraft.origin_vertiport_id,
                                        destination_vertiport_id=aircraft.destination_vertiport_id)
        yield arrival_fix_usage_request
        self.logger.debug(f'Aircraft {aircraft.tail_number} has been assigned to the arrival fix resource at {aircraft.destination_vertiport_id}')

        holding_end = self.env.now
        aircraft.holding_time = holding_end - holding_start
        aircraft.update_holding_energy_consumption(aircraft.holding_time)
        aircraft.save_process_time(event='holding', process_time=aircraft.holding_time)

        self.event_saver.save_aircraft_holding_time(vertiport_id=aircraft.destination_vertiport_id,
                                                    waiting_time=aircraft.holding_time)        

        aircraft.arrival_fix_resource = arrival_fix_resource
        aircraft.arrival_fix_usage_request = arrival_fix_usage_request  

        yield self.env.process(self.fato_and_parking_pad_usage_process(aircraft=aircraft))       

    def fato_and_parking_pad_usage_process(self, aircraft: object):
        if aircraft.parking_space_id is None and \
            aircraft.assigned_fato_id is None and \
                self.vertiports[aircraft.destination_vertiport_id].vertiport_layout.num_parking_pad > 0:
            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft))
            # Save flight_direction
            flight_direction = aircraft.flight_direction
            # Landing process
            yield self.env.process(self.simulate_landing_process(aircraft=aircraft))
            # Taxi
            yield self.env.process(self.simulate_taxi_process(aircraft=aircraft, operation_type='arrival'))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft)    

            # print(f"Aircraft {aircraft.tail_number} parked with passengers: {[p.passenger_id for p in aircraft.passengers_onboard]} to {aircraft.destination_vertiport_id} at time : {miliseconds_to_hms(self.env.now)}. Available aircraft at that location: {[a.tail_number for a in self.vertiports[aircraft.destination_vertiport_id].available_aircraft_store.items]}")

            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)
            self.save_passenger_trip_times(aircraft=aircraft, flight_direction=flight_direction)

            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
       
        # Only FATO case and starting point is not a FATO
        elif aircraft.parking_space_id is None and aircraft.assigned_fato_id is None:
            # NOTE: We don't request FATO here because it will be requested in the landing_process. However
            # in the case of single FATO, we are increasing the FATO reservation time from 
            # time_descend_transition + time_hover_descend to time_descend_transition + time_hover_descend + time_descend
            # yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival'))  

            # Landing process
            yield self.env.process(self.simulate_landing_process(aircraft=aircraft)) 
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)   
            self.save_passenger_trip_times(aircraft=aircraft)
                   
            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Charging process
            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))         

        # If the aircraft is not assigned to a parking pad but its starting location is a FATO
        elif aircraft.parking_space_id is None and \
            aircraft.assigned_fato_id is not None and \
                self.vertiports[aircraft.destination_vertiport_id].vertiport_layout.num_parking_pad > 0:
            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft))
            # Get FATO
            yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival'))
            # Taxi
            yield self.env.process(self.simulate_taxi_process(aircraft=aircraft, operation_type='arrival'))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Charging process
            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
 
        
        # Only FATO case and the starting location is a FATO
        elif aircraft.assigned_fato_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get FATO
            yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival', fato_id=aircraft.assigned_fato_id))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()

            if aircraft.initial_process in ['charging', None]:
                # Charging process
                yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
                aircraft.initial_process = None
            elif aircraft.initial_process == 'parking':
                # Put aircraft into available aircraft store
                self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)
                aircraft.initial_process = None
            else:
                raise ValueError('Unknown initial process for the aircraft')            
            
        # If the aircraft is assigned to a parking pad
        elif aircraft.parking_space_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft, parking_space_id=aircraft.parking_space_id))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            
            if aircraft.initial_process in [AircraftStatus.CHARGE, None]:
                # Charging process
                yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
                aircraft.initial_process = None
            elif aircraft.initial_process == AircraftStatus.IDLE:
                # Put aircraft into available aircraft store
                self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)
                aircraft.initial_process = None
            else:
                raise ValueError('Unknown initial process for the aircraft')                 
            
    def aircraft_charging_process(self, aircraft: object):

        # Charge the aircraft
        yield self.env.process(
            aircraft.charge_aircraft(
                parking_space=aircraft.parking_space,
                shared_charger=self.vertiports[aircraft.destination_vertiport_id].shared_charger_sets
            )
        )

        if self.sim_params['only_aircraft_simulation']:
            yield self.env.process(
                self.simulate_aircraft_departure_process(aircraft=aircraft, origin_vertiport_id=aircraft.origin_vertiport_id, destination_vertiport_id=None)
            )                 