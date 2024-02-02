import simpy
from typing import Any, Dict, List
from .utils.units import ms_to_min, sec_to_ms
from .utils.get_state_variables import get_simulator_states
from .utils.helpers import get_passenger_id_str_from_passenger_list, careful_round, filter_nested_dict_by_given_keys
from .run_step_by_step_simulation import get_user_input, fast_forward_simulation, modify_aircraft_state
import numpy as np
import gymnasium as gym
from collections import defaultdict

class SimEngineRunner:
    def __init__(self, 
                 env: simpy.Environment, 
                 sim_setup: object,
                 stopping_events: Dict = None,
                 reward_function_parameters: Dict = None,
                 terminal_event: object = None,
                 truncation_event: object = None):
        self.env = env
        self.sim_setup = sim_setup
        self.stopping_events = stopping_events
        self.max_sim_time = sec_to_ms(self.sim_setup.sim_params['max_sim_time'])
        self.terminal_event = terminal_event
        self.truncation_event = truncation_event
        self.reward_function_parameters = reward_function_parameters
                        
    def run_uninterrupted_simulation(self):
        """
        Run the simulation until the max simulation time or there is no event left in the queue.
        """
        if self.sim_setup.sim_mode['offline_optimization']:
            self.env.run()
        else:
            self.env.run(until=self.max_sim_time)

    def apply_action(self, action: gym.spaces.Tuple):
        """
        Apply action to the simulation.

        Actions:
        0: Do nothing
        1: Idle to charge
        2: Idle to service (vertiport 1)
        3: Idle to service (vertiport 2)
        """
        process_list = []
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if action[aircraft_id] == 1:
                self.env.process(aircraft.charge_aircraft(parking_space=aircraft.parking_space))
            elif action[aircraft_id] == 2:
                self.env.process(self.sim_setup.system_manager.reserve_aircraft(aircraft))
            process_list.append(aircraft.process)
        yield self.env.all_of(process_list)
    
    def get_current_state(self):
        """
        Get the current state of the simulation.
        """
        return get_simulator_states(vertiports=self.sim_setup.vertiports,
                                    aircraft_agents=self.sim_setup.system_manager.aircraft_agents,
                                    num_initial_aircraft=self.sim_setup.system_manager.num_initial_aircraft,
                                    simulation_states=self.sim_setup.sim_params['simulation_states'])
    
    def get_initial_state(self):
        """
        Reset the simulation.
        """
        while self.sim_setup.system_manager.get_aircraft_count() < self.sim_setup.system_manager.num_initial_aircraft:
            self.env.step()   
        return self.get_current_state()
    
    def calculate_reward(self):
        """
        Calculate the reward for the current state of the simulation.
        """
        return sum(
            vertiport.get_total_passenger_waiting_time()
            for _, vertiport in self.sim_setup.vertiports.items()
        )
    
    def advance_to_next_stopping_event(self):
        """
        Run the simulation until the next defined stopping event.
        """
        while self._should_continue_simulation():
            self.env.step()
            # Check if any events in stopping_events have triggered.
            for event_name, event in self.stopping_events.items():
                if event.triggered:
                    # Get the states
                    states = self.get_current_state()
                    # Reset the event
                    self._reset_event(event_name)
                    # Return the states
                    return states, self.calculate_reward(), self.check_terminated(), self.check_truncated()

        self._handle_end_of_simulation()
        return self.get_current_state(), self.calculate_reward(), self.check_terminated(), self.check_truncated()

    def _should_continue_simulation(self):
        return (self.env.peek() < self.max_sim_time and 
                not self.terminal_event.triggered and 
                not self.truncation_event.triggered)

    def _reset_event(self, event_name):
        self.stopping_events[event_name] = self.env.event()

    def _handle_end_of_simulation(self):
        if self.env.peek() < self.max_sim_time:
            self.terminal_event.succeed()

    def check_terminated(self):
        """
        Check if the simulation has terminated.
        """
        return self.terminal_event.triggered
    
    def check_truncated(self):
        """
        Check if the simulation has truncated.
        """
        return self.truncation_event.triggered