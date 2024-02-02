from .base_system_manager import BaseSystemManager
from typing import Any, List, Union, Dict
import simpy
from ..utils.helpers import get_passenger_ids_from_passenger_list, miliseconds_to_hms, duplicate_str, \
    get_random_process_id, calculate_passenger_consolidation_time, check_whether_node_exists, careful_round, flatten_dict, write_to_db
from ..aircraft.aircraft import AircraftStatus
from ..utils.units import sec_to_ms, ms_to_sec, ms_to_min, miles_to_m



class OfflineOptimizationSystemManager(BaseSystemManager):
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
                 sim_mode: str = 'offline_optimization',
                 external_optimization_params: Dict = None):
        
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

        self.external_optimization_params = external_optimization_params
        self.aircraft_agents = {}
        self.passenger_agents = {}
        self.taxi_resource = simpy.Resource(self.env, 1)
        self.aircraft_battery_models = self.build_aircraft_battery_models()           


    def retrieve_aircraft(self, origin_vertiport_id, tail_number=None, soc=None):
        """
        Retrieve an aircraft from a specific vertiport, specific tail_number, specific soc.

        Args:
            origin_vertiport_id: The ID of the vertiport to retrieve the aircraft from.
            tail_number: The tail number of the aircraft to retrieve (default is None).
            soc: The State of Charge (SOC) of the aircraft to retrieve (default is None).

        Returns:
            The requested aircraft, or None if no suitable aircraft was found.
        """
        # print('Searching for aircraft...')
        # Handle the case where neither tail_number nor soc are specified
        if tail_number is None and soc is None:
            aircraft_request = self.vertiports[origin_vertiport_id].available_aircraft_store.get()
            return aircraft_request # (yield aircraft_request)

        # Handle the case where only soc is specified
        if tail_number is None and soc is not None:
            self.validate_soc(soc)

            return self.vertiports[origin_vertiport_id].available_aircraft_store.get(
                lambda aircraft: abs(soc - aircraft.soc) <= self.external_optimization_params['charge_assignment_sensitivity'])
        

    def validate_tail_number(self, tail_number):
        if not isinstance(tail_number, (int, str)):
            raise ValueError("Tail numbers should be a string or an integer.")

    def validate_soc(self, soc):
        if not isinstance(soc, (int, float)):
            raise ValueError("SOC should be an integer or a float.")
        

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
            
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)
            self.save_passenger_trip_times(aircraft=aircraft, flight_direction=flight_direction)

            if aircraft.flight_duration and self.env.now - aircraft.pushback_time < self.external_optimization_params['flight_duration_constant']:
                ground_holding_time = self.external_optimization_params['flight_duration_constant'] - (self.env.now - aircraft.pushback_time) - 1
                self.logger.debug(f"|{duplicate_str(aircraft.tail_number)}| Flight duration is less than {ms_to_min(self.external_optimization_params['flight_duration_constant'])} minutes."
                                    f' Pushback time was {miliseconds_to_hms(aircraft.pushback_time)}. Aircraft {aircraft.tail_number}'
                                    f' will hold for {ms_to_min(ground_holding_time)} ({miliseconds_to_hms(ground_holding_time)}) mins at {aircraft.location}')
                yield self.env.timeout(ground_holding_time)
            else:
                ground_holding_time = 0
            aircraft.ground_holding_end_time = self.env.now
            aircraft.idle_time += ground_holding_time
            aircraft.save_process_time(event='idle', process_time=ground_holding_time)
            self.logger.info(f'Saved idle time of {miliseconds_to_hms(ground_holding_time)} for ground holding for aircraft {aircraft.tail_number} at {aircraft.location}.')
            self.logger.debug(f'|{duplicate_str(aircraft.tail_number)}| Aircraft {aircraft.tail_number} is now ready for allocation.')
            aircraft.detailed_status = 'idle'
            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.current_vertiport_id,
                                                            aircraft=aircraft)           

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

            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft)

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

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}')

            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)    
        
        # Only FATO case and the starting location is a FATO
        elif aircraft.assigned_fato_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}')

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

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}')

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

        # Put aircraft into available aircraft store
        self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                        aircraft=aircraft)

    def simulate_passenger_arrival(self, passenger):
        """ Simulate passenger."""
        # Save passenger creation time
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='vertiport_entrance')
        passenger.vertiport_arrival_time = self.env.now

        # Put the passenger into the waiting room
        self.put_passenger_into_waiting_room(passenger)
        yield self.env.timeout(0)

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
