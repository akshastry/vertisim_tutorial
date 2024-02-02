# from utils.get_state_variables import get_human_readable_state_variables, get_machine_readable_state_variables, get_simulator_states
from .utils.get_state_variables import get_simulator_states, get_human_readable_state_variables
import time
from typing import Dict, List
import json

def run_step_by_step_simulation(env, vertiports, airspace, structural_entity_groups, system_manager, max_sim_time):
    while env.peek() < max_sim_time:
        # Run the simulation one step.
        env.step()
        # Get the human-readable states
        get_human_readable_simulation_state_variables(vertiports=vertiports, airspace=airspace, system_manager=system_manager)
        # Wait for user input
        modify = get_user_input()
        # If the input is a number, then fast-forward the simulation by the entered number of steps.
        if modify.isdigit():
            fast_forward_simulation(env=env, steps=int(modify))
        # If the user wants to modify an aircraft's state, then modify it.
        elif modify.lower() == "y":
            modify_aircraft_state(env=env, structural_entity_groups=structural_entity_groups, airspace=airspace, system_manager=system_manager)            


def run_steps_until_specific_events(env: object, 
                                    max_sim_time: int, 
                                    vertiports: Dict,
                                    airspace: object,
                                    system_manager: object,
                                    simulation_states: Dict[str, List], # TODO: Now simulation states are a dict of lists.
                                    terminal_event: object,
                                    stopping_events: Dict):
    while env.peek() < max_sim_time and terminal_event.triggered == False:
        env.step()
        # Check if any of the stopping events have occurred.
        for event_name, event in stopping_events.items():
            if event.triggered:
                # Get the states
                states = get_simulator_states(vertiports, airspace, system_manager, simulation_states)
                print(states.json(indent=4))
                # Reset the event
                stopping_events[event_name] = env.event()
                # Return the states


def get_human_readable_simulation_state_variables(vertiports, airspace, system_manager):
    get_human_readable_state_variables(vertiports=vertiports, 
                        airspace_states=airspace.get_airspace_states_dict(),
                        aircraft_agents=system_manager.aircraft_agents,
                        passenger_agents=system_manager.passenger_agents)
    
def call_simulator_states(vertiports, airspace, system_manager, simulation_states):
    get_simulator_states(vertiports=vertiports, 
                        aircraft_agents=system_manager.aircraft_agents,
                        num_aircraft=system_manager.num_aircraft,
                        simulation_states=simulation_states)    

def get_user_input():
    return input("Do you want to modify any aircraft's state? (y/n) or fast-forward simulation steps (# simulation steps to fast-forward): ")


def fast_forward_simulation(env, steps):
    for _ in range(steps):
        env.step()


def modify_aircraft_state(env, structural_entity_groups, airspace, system_manager):
    aircraft_id, state, new_value = map(str.strip, input("Enter the aircraft id, state, and new value (comma separated): ").split(','))
    if aircraft_id in system_manager.aircraft_agents:
        try_modify_state(env, aircraft_id, state, new_value, structural_entity_groups, airspace, system_manager)
    else:
        print("Invalid aircraft id.")
        print("Valid aircraft ids are: ", list(system_manager.aircraft_agents.keys()))


def try_modify_state(env, aircraft_id, state, new_value, structural_entity_groups, airspace, system_manager):
    aircraft = system_manager.aircraft_agents[aircraft_id]
    if hasattr(aircraft, state):
        update_state(env, aircraft, state, new_value, structural_entity_groups, airspace, system_manager)
    else:
        print(f"Aircraft does not have a state named '{state}'")


def update_state(env, aircraft, state, new_value, structural_entity_groups, airspace, system_manager):
    try:
        new_value = convert_new_value(state, new_value)
        setattr(aircraft, state, new_value)
        update_location_state(env, aircraft, state, new_value, structural_entity_groups, airspace, system_manager)
        print(f"Successfully modified aircraft {aircraft.tail_number}'s state '{state}' to {new_value}")
    except ValueError as e:
        print(e)


def convert_new_value(state, new_value):
    if state == 'soc':
        return int(new_value)
    elif state in ['speed', 'priority', 'serviced_time_at_the_location']:
        return float(new_value)
    else:
        return new_value


def update_location_state(env, aircraft, state, new_value, structural_entity_groups, airspace, system_manager):
    if state == 'location':
        if new_value in structural_entity_groups['fato']:
            update_aircraft_fato(env, aircraft, new_value, system_manager)
        elif new_value in structural_entity_groups['parking_pad']:
            update_aircraft_parking_pad(env, aircraft, new_value, system_manager)
        elif new_value in list(airspace.airlink_resources[aircraft.flight_direction].keys())[:-1]:
            update_aircraft_location(env, aircraft, new_value)


def update_aircraft_fato(env, aircraft, new_value, system_manager):
    aircraft.assigned_fato_id = new_value
    env.process(system_manager.fato_and_parking_pad_usage_process(aircraft=aircraft))

def update_aircraft_parking_pad(env, aircraft, new_value, system_manager):
    aircraft.parking_space_id = new_value
    env.process(system_manager.fato_and_parking_pad_usage_process(aircraft=aircraft))

def update_aircraft_location(env, aircraft, new_value):
    aircraft.location = new_value
    env.process(aircraft.fly_in_the_airspace(user_modification=True))

