from typing import List, Dict, Set
from ..aircraft.aircraft import AircraftStatus

from ..utils import rl_utils
from ..utils.helpers import (
    current_and_lookahead_pax_count,
    get_waiting_passenger_ids,
    get_total_waiting_passengers_at_vertiport
)


class RewardFunction:
    def __init__(self, config: Dict, reverse_action_dict: Dict, sim_setup: object) -> None:
        self.config = config
        self.reverse_action_dict = reverse_action_dict
        self.sim_setup = sim_setup
        self.reward = 0

    def compute_reward(self, actions: List[int]) -> None:
        """
        Calculate the reward based on the provided actions.
        """
        departing_passengers = self._compute_departing_passengers(actions)
        self._apply_flight_rewards(actions=actions,
                                   departing_passengers=departing_passengers)
        self.reward += self._waiting_time_penalty(departing_passengers=departing_passengers)
        self.reward += self.config['external_optimization_params']['reward_function_parameters']['step_penalty']
        self.reward += self._compute_holding_penalty()

    def _compute_holding_penalty(self) -> float:
        """
        Compute the holding penalty.
        """
        holding_time = 0
        for _, vertiport in self.sim_setup.vertiports.items():
            holding_time += vertiport.get_holding_time_cost()
        return holding_time * self.config['external_optimization_params']['reward_function_parameters']['holding_penalty']

    def _compute_departing_passengers(self, actions: List[int]) -> Set[int]:
        """
        Compute the set of departing passengers.
        """
        departing_passengers = set()
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                departing_passengers.update(get_waiting_passenger_ids(
                    sim_setup=self.sim_setup,
                    exclude_pax_ids=departing_passengers,
                    origin_vertiport_id=aircraft.current_vertiport_id,
                    destination_vertiport_id=self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                ))
        return departing_passengers

    def _apply_flight_rewards(self, actions: List[int], departing_passengers: Set[int]) -> None:
        """
        Apply flight-based rewards and penalties.
        """
        pax_count = len(departing_passengers)
        trip_time_counter = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                dest_vertiport_id = self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                trip_time_counter += self._calculate_trip_time(origin_id=aircraft.current_vertiport_id, destination_id=dest_vertiport_id)
                self._apply_unnecessary_flight_penalty(dest_id=dest_vertiport_id, 
                                                       origin_id=aircraft.current_vertiport_id)
                
                # Penalize if there is an aircraft already headed to the same destination vertiport and
                # that aircraft + existing aircraft is suffient to serve the demand.
                self._penalize_for_not_counting_arriving_aircraft(dest_id=dest_vertiport_id)

            elif actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                self.reward += self._get_charge_reward(aircraft)

            self.reward += self.compute_action_reward_wrt_pax_count(aircraft=aircraft, action=actions[aircraft_id])

        self._apply_trip_and_flight_cost_reward(pax_count, trip_time_counter)

    def _calculate_trip_time(self, origin_id: int, destination_id: int) -> int:
        """
        Calculate the trip time for a flight.
        """
        return self.sim_setup.system_manager.get_mission_length(
            origin_vertiport_id=origin_id,
            destination_vertiport_id=destination_id
        )
    
    def _get_destination_vertiport_id(self, aircraft_id: int, actions: List[int]) -> int:
        """
        Get the destination vertiport id for a given aircraft.
        """
        return self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]
    
    def _penalize_for_not_counting_arriving_aircraft(self, dest_id: int) -> None:
        # Get the aircraft that is actually flying (not on the ground) and is headed to the destination vertiport
        lookahead_aircraft_count = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if aircraft.status == AircraftStatus.FLY and \
                aircraft.flight_direction == f"{aircraft.current_vertiport_id}_{dest_id}" and \
                    aircraft.soc > self.config['aircraft_params']['min_reserve_soc']:
                lookahead_aircraft_count += 1
        lookahead_aircraft_count += self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_id)
        if lookahead_aircraft_count * self.config['aircraft_params']['pax'] >= current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id]):
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']

    def _apply_unnecessary_flight_penalty(self, dest_id: int, origin_id: int) -> None:
        """
        Apply penalty for unnecessary flights and reward for useful repositioning.
        """
        if current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id]) == 0:
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']
        elif current_and_lookahead_pax_count(self.sim_setup.vertiports[origin_id]) == 0:
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

        # If there are available aircraft at the destination vertiport and the aircraft is sufficient to serve the demand
        # then penalize the empty flight.
        # num_aircraft_at_destination = self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_id)
        # num_expected_pax = current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id])
        # if num_aircraft_at_destination > 0 and num_expected_pax <= num_aircraft_at_destination * self.config['aircraft_params']['pax']:
        #     self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']

    def _get_charge_reward(self, aircraft: object) -> float:
        """
        Calculate the reward for charging.
        """
        if aircraft.system_manager.sim_mode == 'rl':
            return rl_utils.charge_reward(
                current_soc=aircraft.soc,
                soc_reward_threshold=aircraft.charging_strategy.soc_reward_threshold
            ) * self.config['external_optimization_params']['reward_function_parameters']['charge_reward']
        return 0

    def _apply_trip_and_flight_cost_reward(self, pax_count: int, trip_time: int) -> None:
        """
        Apply rewards and costs related to trips and flight time.
        """
        self.reward += self.config['external_optimization_params']['reward_function_parameters']['trip_reward'] * pax_count
        self.reward += self.config['external_optimization_params']['reward_function_parameters']['flight_cost'] * trip_time

    def _waiting_time_penalty(self, departing_passengers) -> float:
        """
        Calculate the waiting time penalty.
        """
        waiting_time_cost_type = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_type']
        waiting_time_cost_unit = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_unit']

        return sum(
            vertiport.get_waiting_time_cost(unit=waiting_time_cost_unit,
                                            type=waiting_time_cost_type,
                                            exclude_pax_ids=departing_passengers,
                                            last_decision_time=None)
            for _, vertiport in self.sim_setup.vertiports.items()) * \
            self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost']

    def compute_action_reward_wrt_pax_count(self, aircraft, action):
        """
        Compute the reward or penalty for the given number of waiting passengers. TODO: Needs update
        Danger: The rewards for 'do nothing' and 'charge' when there are no passengers might 
        encourage the agent to avoid repositioning even when it's necessary for future demand.
        """
        num_waiting_pax = get_total_waiting_passengers_at_vertiport(sim_setup=self.sim_setup, vertiport_id=aircraft.current_vertiport_id)
        if num_waiting_pax <= self.config['aircraft_params']['pax']//2 and \
            action == self.reverse_action_dict['CHARGE']:
            return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_charge_reward']
        elif num_waiting_pax <= self.config['aircraft_params']['pax']//2 and \
            action == self.reverse_action_dict['DO_NOTHING']:
            return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_do_nothing_reward']   
        # # TODO: This requires to check the energy cons of the flight and the SoC level of the aircraft.
        # # Also needs to check if there are a
        # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
        #     action == self.reverse_action_dict['CHARGE'] and \
        #     soc >= self.config['aircraft_params']['min_reserve_soc']:
        #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_charge_penalty']
        # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
        #     action == self.reverse_action_dict['DO_NOTHING'] and \
        #         soc >= self.config['aircraft_params']['min_reserve_soc']:
        #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_do_nothing_penalty']
        else:
            return 0                 

    def reset_rewards(self):
        """
        Reset the rewards for the current state of the simulation.
        """
        self.sim_setup.system_manager.trip_counter = 0
        self.sim_setup.system_manager.trip_time_counter = 0
        self.sim_setup.system_manager.truncation_penalty = 0
        # self.self.sim_setup.system_manager.occupancy_reward = 0
        self.sim_setup.system_manager.spill_counter = 0
        self.sim_setup.system_manager.charge_reward = 0
        self.sim_setup.system_manager.holding_time_counter = 0
        self.reward = 0




# from typing import List, Dict

# from ..utils import rl_utils
# from ..utils.helpers import current_and_lookahead_pax_count, get_waiting_passenger_ids, get_total_waiting_passengers_at_vertiport


# class RewardFunction:
#     def __init__(self, config: Dict, reverse_action_dict: Dict, sim_setup: object) -> None:
#         self.reward = 0
#         self.config = config
#         self.reverse_action_dict = reverse_action_dict
#         self.sim_setup = sim_setup

#     def compute_reward(self, actions: List) -> None:
#         """
#         Overarching reward function
#         """
#         pax_count = 0
#         trip_time_counter = 0
#         # Departing pax is a set to avoid double counting passengers on flights originating from aircraft located at the same vertiport.
#         departing_passengers = set()
#         # Iterate through each aircraft and calculate the reward
#         for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():    
#             # If the action value is less than the number of vertiports, it means it's a flight assignment
#             if actions[aircraft_id] < self.sim_setup.num_vertiports():  
#                 # Get the destination vertiport id
#                 dest_vertiport_id = self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]       
#                 # Update the trip time counter
#                 trip_time_counter += self.sim_setup.system_manager.get_mission_length(origin_vertiport_id=aircraft.destination_vertiport_id,
#                                                                                  destination_vertiport_id=dest_vertiport_id)   
#                 # Update the departing passengers set
#                 departing_passengers.update(get_waiting_passenger_ids(origin_vertiport_id=aircraft.destination_vertiport_id,
#                                                                       destination_vertiport_id=dest_vertiport_id))  
#                 # To avoid double counting passengers on flights originating from aircraft located at the same vertiport.
#                 pax_count += len(departing_passengers) - pax_count

#                 # Penalty for unnecessary flight
#                 # Get vertiport object from the destination vertiport id
#                 dest_vertiport = self.sim_setup.vertiports[dest_vertiport_id]
#                 # Get the number of waiting and the lookahead passengers at the destination vertiport
#                 demand = current_and_lookahead_pax_count(dest_vertiport)
#                 if demand == 0:
#                     self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']
                
#                 # Check if there is no demand at the origin vertiport, reward for repositioning if there is a demand at the destination vertiport
#                 if current_and_lookahead_pax_count(aircraft.destination_vertiport_id) == 0 and \
#                     demand > 0:
#                     self.reward += self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

#             elif actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
#                 charge_reward = self.get_charge_reward(aircraft=aircraft)
#                 self.reward += charge_reward * self.config['external_optimization_params']['reward_function_parameters']['charge_reward'] 

#         self.reward += self.config['external_optimization_params']['reward_function_parameters']['trip_reward'] * pax_count
#         self.reward += self.config['external_optimization_params']['reward_function_parameters']['flight_cost'] * trip_time_counter

#         waiting_time_cost_type = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_type']
#         waiting_time_cost_unit = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_unit']

#         self.reward += self.waiting_time_penalty(how=waiting_time_cost_type, 
#                                             unit=waiting_time_cost_unit, 
#                                             exclude_pax_ids=departing_passengers)     
        

#     def waiting_time_penalty(self, how='linear', unit='hour', exclude_pax_ids=[]):
#         """
#         Calculate the waiting time penalty for the current state of the simulation.

#         Parameters
#         ----------
#         how: str
#             The type of the waiting time penalty. Either linear or exponential.
#         """
#         return sum(
#             vertiport.get_waiting_time_cost(unit=unit, 
#                                             type=how,
#                                             exclude_pax_ids=exclude_pax_ids,
#                                             last_decision_time=None)
#             for _, vertiport in self.sim_setup.vertiports.items()) * \
#                 self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost']                               


#     def get_charge_reward(self, aircraft: object) -> None:
#         """
#         Reward for charging
#         """
#         if aircraft.system_manager.sim_mode == 'rl':
#             aircraft.system_manager.charge_reward += rl_utils.charge_reward(
#                 current_soc=aircraft.soc,
#                 soc_reward_threshold=aircraft.charging_strategy.soc_reward_threshold
#                 )
            
#     def compute_action_reward_wrt_pax_count(self, aircraft, action):
#         """
#         Compute the reward or penalty for the given number of waiting passengers. TODO: Needs update
#         Danger: The rewards for 'do nothing' and 'charge' when there are no passengers might 
#         encourage the agent to avoid repositioning even when it's necessary for future demand.
#         """
#         num_waiting_pax = get_total_waiting_passengers_at_vertiport(sim_setup=self.sim_setup, vertiport_id=aircraft.destination_vertiport_id)
#         if num_waiting_pax == 0 and \
#             action == self.reverse_action_dict['CHARGE']:
#             return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_charge_reward']
#         elif num_waiting_pax == 0 and \
#             action == self.reverse_action_dict['DO_NOTHING']:
#             return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_do_nothing_reward']   
#         # # TODO: This requires to check the energy cons of the flight and the SoC level of the aircraft.
#         # # Also needs to check if there are a
#         # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
#         #     action == self.reverse_action_dict['CHARGE'] and \
#         #     soc >= self.config['aircraft_params']['min_reserve_soc']:
#         #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_charge_penalty']
#         # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
#         #     action == self.reverse_action_dict['DO_NOTHING'] and \
#         #         soc >= self.config['aircraft_params']['min_reserve_soc']:
#         #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_do_nothing_penalty']
#         else:
#             return 0                 

#     def reset_rewards(self):
#         """
#         Reset the rewards for the current state of the simulation.
#         """
#         self.sim_setup.system_manager.trip_counter = 0
#         self.sim_setup.system_manager.trip_time_counter = 0
#         self.sim_setup.system_manager.truncation_penalty = 0
#         # self.self.sim_setup.system_manager.occupancy_reward = 0
#         self.sim_setup.system_manager.spill_counter = 0
#         self.sim_setup.system_manager.charge_reward = 0
#         self.sim_setup.system_manager.holding_time_counter = 0
#         self.reward = 0