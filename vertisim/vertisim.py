import simpy
from typing import Any, Dict, List
from .sim_setup import SimSetup
from .aircraft.aircraft import AircraftStatus
from .utils.helpers import get_stopping_events, get_truncation_events, miliseconds_to_hms, \
    create_action_enum, convert_action_values_to_enum, create_action_dict, reverse_dict, \
        current_and_lookahead_pax_count, get_total_waiting_passengers_at_vertiport, \
        get_inflow_demand_to_vertiport
import time
import numpy as np
import gymnasium as gym
from collections import defaultdict
from pprint import pformat
from enum import Enum
from .utils.get_state_variables import get_simulator_states
from .utils import rl_utils
from .rl_methods.reward_function import RewardFunction
# from .rl_methods.action_mask import ActionMask
from .utils.fetch_data_from_db import fetch_latest_data_from_db
from .initiate_flow_entities import initiate_flow_entities
from .utils.units import sec_to_ms, ms_to_hr, sec_to_min
from .run_step_by_step_simulation import run_step_by_step_simulation, run_steps_until_specific_events


class VertiSim:
    def __init__(self, 
                 env: simpy.Environment, 
                 config: Dict):
        self.env = env
        self.config = config
        self.terminal_event, self.truncation_event, self.truncation_events, self.stopping_events = self.set_simulation_events(env=self.env, config=self.config)

        self.sim_start_time = time.time()
        self.sim_setup = SimSetup(env=self.env,
                                  sim_params=config['sim_params'],
                                  sim_mode=config['sim_mode'],
                                  external_optimization_params=config['external_optimization_params'],
                                  network_and_demand_params=config['network_and_demand_params'],
                                  airspace_params=config['airspace_params'],
                                  passenger_params=config['passenger_params'],
                                  aircraft_params=config['aircraft_params'],
                                  output_params=config['output_params'],
                                  stopping_events=self.stopping_events,
                                  truncation_events=self.truncation_events,
                                  truncation_event=self.truncation_event)    
        # Initiate flow entities and create events
        # initiate_flow_entities(sim_setup=self.sim_setup)       
        # Set the max simulation time
        self.max_sim_time = sec_to_ms(self.sim_setup.sim_params['max_sim_time'])
        # The status of the simulation provides information whether the simulation running or not
        # Remove the old logs.
        # self.sim_setup.logger.remove_logs(seconds=10)

        if self.config['sim_mode']['rl']:
            self.setup_for_rl()

        self.status = True

    def setup_for_rl(self):
        # self.actions = create_action_enum(self.sim_setup.vertiport_ids)        
        self.action_dict = create_action_dict(self.sim_setup.vertiport_ids)
        self.reverse_action_dict = reverse_dict(self.action_dict)
        self.reward_function = RewardFunction(config=self.config, reverse_action_dict=self.reverse_action_dict, sim_setup=self.sim_setup)

    def set_simulation_events(self, env, config):
        """
        Set the simulation events.
        """
        terminal_event = env.event()
        truncation_event = env.event()
        truncation_events = get_truncation_events(env=env,
                                                  truncation_events=config["external_optimization_params"]["truncation_events"],
                                                  aircraft_count=self.get_aircraft_count())
        stopping_events = get_stopping_events(env=env, 
                                              stopping_events=config["external_optimization_params"]["stopping_events"],
                                              aircraft_count=self.get_aircraft_count(),
                                              vertiport_ids=self.get_vertiport_ids(),
                                              pax_count=self.get_passenger_count())
        return terminal_event, truncation_event, truncation_events, stopping_events

    def run(self):
        # This only runs uninterrupted simulation. No distinction for online optimization
        self.run_uninterrupted_simulation()          
        self.finalize_simulation()
        # performance_metrics = self.sim_setup.calculate_performance_metrics()
        # self.sim_setup.save_results(performance_metrics)
        # self.sim_setup.print_passenger_trip_time_stats()
        # self.sim_setup.log_brief_metrics(print_metrics=True)

    def finalize_simulation(self):
        if not self.sim_setup.sim_mode['rl']:
            self.print_total_time_to_run()
            if self.env.now > 0:
                performance_metrics = self.sim_setup.calculate_performance_metrics()
                self.sim_setup.save_results(performance_metrics)

    def run_uninterrupted_simulation(self):
        """
        Run the simulation until the max simulation time or there is no event left in the queue.
        """
        if self.sim_setup.sim_mode['offline_optimization']:
            self.env.run()
        else:
            self.env.run(until=self.max_sim_time)

    def reset(self):
        """
        Reset the simulation.
        """
        if not self.terminal_event.triggered:
            self.sim_setup.logger.finalize_logging()
            return self.terminal_event.succeed()
    
    def step(self, actions):
        """
        Run the simulation for one timestep.
        """
        # Get the number of triggers for this step
        total_triggers = sum(action != 0 for action in actions)
        # Apply the actions and rewards at the same time to keep the Markov property
        self.env.process(self.apply_action_reward(actions=actions))
        # Advance to the next stopping event. The returned state and reward pair should be
        # the state and reward after all the actions are applied.
        if self.config['external_optimization_params']['periodic_time_step']:
            return self.advance_periodically()
        else:
            return self.advance_to_next_stopping_event(total_triggers)

    def apply_action_reward(self, actions: Enum):
        """
        Apply action to the simulation and compute the reward before execution.
        We change the state of the aircrafts based on the action before we step the simulation
        to be able to calculate the reward at the same time for the given action list.

        Actions:
        0, 1, ..., N-1: Idle to fly to vertiport n
        N: Idle to charge
        N+1: Do nothing
        """
        self.sim_setup.logger.warning(f"Action: {[self.action_dict[action] for action in actions]}")

        process_list = []
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            # Charge the aircraft
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                aircraft.status = AircraftStatus.CHARGE
                aircraft.is_process_completed = False
            # Do nothing
            elif actions[aircraft_id] == self.reverse_action_dict['DO_NOTHING']:
                aircraft.is_process_completed = True
            # Fly to the vertiport
            elif actions[aircraft_id] < self.sim_setup.num_vertiports():
                aircraft.status = AircraftStatus.FLY  
                aircraft.is_process_completed = False
                aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
                aircraft.origin_vertiport_id = aircraft.current_vertiport_id
                aircraft.destination_vertiport_id = self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]
            else:
                raise ValueError(f"Invalid action: {actions[aircraft_id]}")
                
        # Compute the reward
        self.reward_function.compute_reward(actions=actions)

        # Apply the actions
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                process_list.append(self.env.process(aircraft.charge_aircraft(parking_space=aircraft.parking_space)))
            elif actions[aircraft_id] < self.sim_setup.num_vertiports():
                destination_vertiport_id = self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]
                # Get the departing passengers
                departing_passengers = self.sim_setup.scheduler.collect_departing_passengers_by_od_vertiport(
                    origin_vertiport_id=aircraft.current_vertiport_id, 
                    destination_vertiport_id=destination_vertiport_id)   
                process_list.append(self.env.process(self.sim_setup.system_manager.reserve_aircraft(aircraft=aircraft, 
                                                                                                    destination_vertiport_id=destination_vertiport_id,
                                                                                                    departing_passengers=departing_passengers)))
        yield self.env.all_of(process_list)    

    def action_mask(self, initial_state=False, final_state=False):
        """
        Create the action mask for the current state of the simulation.
        [Fly, Charge, Do nothing]
        """
        mask = []
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():

            current_vertiport_id = aircraft.current_vertiport_id
            current_vertiport = self.sim_setup.vertiports[current_vertiport_id]
            current_vertiport_index = self.sim_setup.vertiport_id_to_index_map[current_vertiport_id]  

            can_do_any = (aircraft.status == AircraftStatus.IDLE and 
                          aircraft.soc <= 100-self.config['external_optimization_params']['soc_increment_per_charge_event'] and 
                          aircraft.soc >= self.config['aircraft_params']['min_reserve_soc'])  
            
            can_fly_and_do_nothing = (aircraft.status == AircraftStatus.IDLE and
                                      aircraft.soc > 100-self.config['external_optimization_params']['soc_increment_per_charge_event'])
            
            # Can do any action
            if can_do_any:
                destination_mask = [1 if destination != current_vertiport_id
                                    else 0 for destination in self.sim_setup.vertiport_ids]
                # aircraft_mask = destination_mask + [1, 1]
                non_flight_actions = [1, 1]
            # Can only charge
            elif aircraft.status == AircraftStatus.IDLE \
                    and aircraft.soc <= self.config['aircraft_params']['min_reserve_soc']:
                destination_mask = [0] * self.sim_setup.num_vertiports()
                # aircraft_mask = destination_mask + [1, 0]
                non_flight_actions = [1, 0]
            # Can fly and do nothing
            elif can_fly_and_do_nothing:
                destination_mask = [1 if destination != current_vertiport_id 
                                    else 0 for destination in self.sim_setup.vertiport_ids] 
                # aircraft_mask = destination_mask + [0, 1] 
                non_flight_actions = [0, 1]         
            # Can do nothing  
            elif aircraft.status == AircraftStatus.CHARGE or aircraft.status == AircraftStatus.FLY:
                destination_mask = [0] * self.sim_setup.num_vertiports()
                # aircraft_mask = destination_mask + [0, 1]     
                non_flight_actions = [0, 1]         
            
            # If flight actions are enabled, check for further conditions
            if sum(destination_mask) != 0:

                # Check if there is no demand at the aircraft's vertiport and it's not over capacity
                if current_and_lookahead_pax_count(current_vertiport) == 0 and current_vertiport.get_parking_pad_availability() > 0:
                    destination_mask = [0] * self.sim_setup.num_vertiports()

                # Check if there are enough idling aircraft at the destination, no passengers, and no holding aircraft at the current vertiport
                for dest_vertiport_id in self.sim_setup.vertiport_ids:
                    if dest_vertiport_id == current_vertiport_id:
                        continue
                    # Get the destination vertiport object
                    dest_vertiport = self.sim_setup.vertiports[dest_vertiport_id]
                    # Get the index of the destination vertiport
                    dest_vertiport_index = self.sim_setup.vertiport_id_to_index_map[dest_vertiport_id]
                    # Get the number of expected passengers at the destination vertiport
                    expected_lookahead_dest_demand = current_and_lookahead_pax_count(dest_vertiport)
                    # # Get the number of expected inflow demand to the destination vertiport
                    # expected_lookahead_inflow_demand = get_inflow_demand_to_vertiport(
                    #     destination_vertiport_id=vertiport_id, 
                    #     system_manager=self.sim_setup.system_manager)   
                    # Get the demand from origin vertiport to the destination vertiport
                    dest_demand_from_origin = current_vertiport.get_waiting_passenger_count()[dest_vertiport_id]
                    # Get the number of aircraft at the destination vertiport                
                    num_aircraft = self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_vertiport_id)
                    # Check if there are enough idling aircraft at the destination vertiport, and no passenger at the current vertiport
                    if num_aircraft * self.config['aircraft_params']['pax'] >= expected_lookahead_dest_demand and \
                        dest_demand_from_origin == 0 and \
                            current_vertiport.num_holding_aircraft == 0:
                        # Mask out the flight action to the destination vertiport
                        destination_mask[dest_vertiport_index] = 0

                if current_vertiport.get_parking_pad_availability() == 0 and \
                        (can_do_any or can_fly_and_do_nothing):
                    # Enable flight action to all vertiports except the current one
                    destination_mask = [1 if idx != current_vertiport_index else 0 for idx in range(self.sim_setup.num_vertiports())]

            # Add the destination mask to the mask
            mask.extend(destination_mask)
            # Add the non-flight actions to the mask
            mask.extend(non_flight_actions)
        

        if initial_state:
            # Set the flight actions to 0 for all aircrafts
            for i in range(0, len(mask), self.sim_setup.num_vertiports() + 2):
                mask[i:i+self.sim_setup.num_vertiports()] = [0] * self.sim_setup.num_vertiports()
        
        if final_state:
            # Only do nothing for all aircraft
            for i in range(0, len(mask), self.sim_setup.num_vertiports() + 2):
                mask[i:i+self.sim_setup.num_vertiports()] = [0] * self.sim_setup.num_vertiports()
                mask[i+self.sim_setup.num_vertiports()+1] = 1
                    
        # Flatten the mask
        return np.array(mask).flatten().tolist()     

    def schedule_periodic_stop_event(self):
        """
        Schedule a stopping event at a fixed interval.
        """
        interval = sec_to_ms(self.config['external_optimization_params']['periodic_time_step'])
        stop_event = self.env.event()
        self.env.process(self.trigger_stop_event_after_interval(stop_event, interval))
        return stop_event

    def trigger_stop_event_after_interval(self, event, interval):
        """
        Process to trigger a given event after a specified interval.
        """
        yield self.env.timeout(interval)
        event.succeed()

    def advance_periodically(self):
        """
        Advance the simulation for a fixed interval and return the state and reward.
        """
        # Schedule the simulation to advance for the interval
        stop_event = self.schedule_periodic_stop_event()
        # Advance the simulation until the stop event is triggered
        while self._should_continue_simulation():
            # Step the simulation
            self.env.step()

            if stop_event.triggered:
                self.sim_setup.logger.warning(f"Periodic time step: {miliseconds_to_hms(self.env.now)}")
                # Return the states
                reward = round(self.reward_function.reward, 2)
                self.reward_function.reset_rewards()

                self.sim_setup.logger.warning(f"Reward: {reward}")
                self.sim_setup.logger.warning(f"Current state: {pformat(self.get_current_state())}")
                self.sim_setup.logger.warning(f"Action mask: {self.action_mask()}")
                return self.get_current_state(), reward, self.is_terminated(), self.is_truncated(), self.action_mask()
            
        return self.get_current_state(), round(self.reward_function.reward, 2), self.is_terminated(), self.is_truncated(), self.action_mask(final_state=True)    


    def advance_to_next_stopping_event(self, total_triggers):
        """
        Runs the simulation until the next defined stopping event.
        Rewards the agent right after the each action is completed.
        Rewards the agent separately for each action.
        """

        while self._should_continue_simulation():
            # Step the simulation
            self.env.step()
          
            # If there are any triggered stopping event queue, pop it out, reset it and return the states
            if len(self.sim_setup.system_manager.triggered_stopping_event_queue):
                # Pop the event on top of the queue
                event_name = self.sim_setup.system_manager.triggered_stopping_event_queue.pop()
                self.sim_setup.logger.warning(f"Triggered event: {event_name}")
                # Reset the event
                self._reset_event(event_name)
                # Return the states
                reward = round(self.reward_function.reward, 2)
                self.reward_function.reset_rewards()

                self.sim_setup.logger.warning(f"Reward: {reward}")
                self.sim_setup.logger.warning(f"Current state: {pformat(self.get_current_state())}")
                self.sim_setup.logger.warning(f"Action mask: {self.action_mask()}")
                return self.get_current_state(), reward, self.is_terminated(), self.is_truncated(), self.action_mask()

        self.finalize_simulation()
        return self.get_current_state(), round(self.reward_function.reward, 2), self.is_terminated(), self.is_truncated(), self.action_mask(final_state=True)    

    def _should_continue_simulation(self):
        return (self.env.peek() < self.max_sim_time and 
                (not self.terminal_event.triggered and 
                not self.truncation_event.triggered and
                not self._check_demand_satisfied()))
    
    def _check_demand_satisfied(self):
        # If system_manager.passenger_arrival_complete == True and if there is no passenger left in the waiting rooms of the vertiports, succeed the terminal event
        # self.sim_setup.logger.warning(f"Is passenger arrival complete: {self.sim_setup.system_manager.passenger_arrival_complete}")
        # self.sim_setup.logger.warning(f"Is all waiting rooms empty: {self.sim_setup.system_manager.check_all_waiting_rooms_empty()}")
        # self.sim_setup.logger.warning(f"Is all passenger travelled: {self.sim_setup.system_manager.is_all_passenger_travelled()}")
        # self.sim_setup.logger.warning(f"Total travelled passengers: {self.sim_setup.system_manager.trip_counter_tracker}")
        if bool(
            self.sim_setup.system_manager.passenger_arrival_complete
            and self.sim_setup.system_manager.check_all_waiting_rooms_empty()
            and self.sim_setup.system_manager.is_all_passenger_travelled()):

            # self.sim_setup.logger.warning("Demand satisfied")
            self.sim_setup.logger.error(self.sim_setup.log_brief_metrics())
            self.terminal_event.succeed()
            return True
        return False
    
    def _reset_event(self, event_name):
        """
        Reset the event by creating a new event.
        """
        self.stopping_events[event_name] = self.env.event()

    def _handle_end_of_simulation(self):
        if self.env.peek() < self.max_sim_time and not self.terminal_event.triggered:
            self.terminal_event.succeed()

    def is_terminated(self):
        """
        Check if the simulation has terminated.
        """
        return self.terminal_event.triggered
    
    def is_truncated(self):
        """
        Check if the simulation has truncated.
        """
        return self.truncation_event.triggered            

    def get_current_state(self):
        """
        Get the current state of the simulation.
        """
        return get_simulator_states(vertiports=self.sim_setup.vertiports,
                                    aircraft_agents=self.sim_setup.system_manager.aircraft_agents,
                                    num_initial_aircraft=self.sim_setup.system_manager.num_initial_aircraft,
                                    simulation_states=self.config['sim_params']['simulation_states'],
                                    reward_function_parameters=self.config['external_optimization_params']['reward_function_parameters'])
    
    def get_initial_state(self):
        """
        Reset the simulation.
        """
        # print("Gettting the instance initial state")
        while self.sim_setup.system_manager.get_available_aircraft_count() < self.sim_setup.system_manager.num_initial_aircraft:          
            self.env.step()
            for event_name, event in self.stopping_events.items():
                if event.triggered:  
                    # Reset the event
                    self._reset_event(event_name) 
                    # Pop them out from the triggered event queue
                    self.sim_setup.system_manager.triggered_stopping_event_queue.pop()
        return self.get_current_state()

    @staticmethod
    def multidiscrete_to_discrete(action, n_actions=3):
        """Converts a MultiDiscrete action to a Discrete action."""
        discrete_action = 0
        for i, a in enumerate(reversed(action)):
            discrete_action += a * (n_actions ** i)
        return discrete_action

    @staticmethod
    def discrete_to_multidiscrete(action, dimensions=4, n_actions=3):
        """Converts a Discrete action back to a MultiDiscrete action."""
        multidiscrete_action = []
        for _ in range(dimensions):
            multidiscrete_action.append(action % n_actions)
            action = action // n_actions
        return list(reversed(multidiscrete_action))   
    

    # def set_action_space(self, config: Dict) -> gym.spaces.Discrete:
    #     """
    #     Creates the action space for each aircraft.
    #     """
    #     n_actions = self.get_action_count()
    #     # n_aircraft = self.get_aircraft_count(config=config)
    #     # return gym.spaces.MultiDiscrete([n_actions] * n_aircraft)
    #     return gym.spaces.Discrete(n_actions)
    
    # def set_observation_space(self) -> gym.spaces.Box:
    #     # Vertiport states
    #     # Get the number of vertiports
    #     num_vertiports = len(self.config['network_and_demand_params']['vertiports'])

    #     # Get number of aircraft defined in the config file
    #     num_aircraft = self.get_aircraft_count()

    #     # Get the total number of state variables
    #     total_state_variables = num_vertiports * self.get_vertiport_state_variable_count() +\
    #                             num_aircraft * self.get_aircraft_state_variable_count() +\
    #                             num_vertiports * self.get_environmental_state_variable_count()
        
    #     # Return the observation space
    #     return gym.spaces.Box(low=0, high=np.inf, shape=(total_state_variables,), dtype=np.float32)
    
    def get_aircraft_count(self) -> int:
        return sum(
            vertiport['aircraft_arrival_process']['num_initial_aircraft_at_vertiport']
            for _, vertiport in self.config['network_and_demand_params']['vertiports'].items()
        )

    def get_vertiport_count(self) -> int:
        return len(self.sim_setup.vertiport_ids)
    
    def get_vertiport_ids(self) -> list:
        return list(self.config['network_and_demand_params']['vertiports'].keys())
    
    def get_action_count(self):
        return self.config['external_optimization_params']['num_actions']

    def get_vertiport_state_variable_count(self):
        return self.config['external_optimization_params']['num_vertiport_state_variables']

    def get_aircraft_state_variable_count(self):
        return self.config['external_optimization_params']['num_aircraft_state_variables']
    
    def get_environmental_state_variable_count(self):
        return self.config['external_optimization_params']['num_environmental_state_variables']
    
    def get_additional_state_variable_count(self):
        return self.config['external_optimization_params']['num_additional_state_variables']
    
    def get_passenger_count(self):
        return sum(
            vertiport['passenger_arrival_process']['num_passengers']
            for _, vertiport in self.config['network_and_demand_params']['vertiports'].items()
        )
    
    def get_num_waiting_passengers_per_vertiport(self, vertiport_id):
        """
        Check the number of waiting passengers at the given vertiport.
        """
        return self.sim_setup.vertiports[vertiport_id].get_waiting_passenger_count()

    def print_total_time_to_run(self):
        end_time = time.time()
        time_taken = end_time - self.sim_start_time
        print(f'Simulation Completed. Total time to run: {round(time_taken, 2)} seconds\n')   
    
    def get_performance_metrics(self):
        return fetch_latest_data_from_db(db_path="sqlite/db/vertisimDatabase.sqlite")