import random
import math
import numpy as np
import os
from enum import Enum
from ..utils.units import sec_to_ms, ms_to_hr, ms_to_sec
from typing import List, Union, Dict, Any, Optional
import re
import pandas as pd # For type hinting
import pkg_resources


def set_seed(seed=None, how='instant'):
    """
    Sets the seed for random number generation.
    :param seed: an optional seed value (int)
    :return: None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    elif how == 'instant':
        seed= int(time.time() % 1 *1000000)
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(42)
        np.random.seed(42)


def is_none(value):
    return value is None


def get_absolute_path(resource_path: str) -> str:
    return pkg_resources.resource_filename('vertisim', resource_path)

def compute_interarrival_times_from_schedule(schedule):
    """
    Computes interarrival times from a schedule.
    :param schedule:
    :return:
    """
    return np.diff(schedule, prepend=0)


def cumulative_sum_array(array: Union[np.ndarray, List]) -> np.ndarray:
    """
    Cumulative sum of an array.
    :param array:
    :return: np.ndarray
    """
    return np.cumsum(array)


def flatten(t):
    """
    Flatten a nested list structure.
    :param t:
    :return:
    """
    return [item for sublist in t for item in sublist]


def get_key_from_value(dictionary, search_val):
    """
    Get key from value in a dictionary.
    :param dictionary:
    :param search_val:
    :return:
    """
    return [key for key, value in dictionary.items() if search_val == value][0]


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_key(my_dict, key):
    """
    Function that returns None if the key is not found, if found return value
    :param my_dict:
    :param key:
    :return:
    """
    return my_dict[key] if key in my_dict else None


def none_checker(val):
    """
    Checks if a value is None. If None returns None, else returns the value.
    :param val:
    :return:
    """
    return None if val is None else val


def pick_random_file(folder_path: str):
    set_seed(how='instant')
    """
    Pick a random file from a folder.
    :param folder_path:
    :return:
    """
    return random.choice(os.listdir(folder_path))


def timestamp_to_datetime(sim_start_time, sim_time):
    """
    Converts a timestamp to a datetime.
    :param sim_start_time:
    :param sim_time:
    :return:
    """
    # date_time = datetime.fromtimestamp(current_timestamp)
    # return date_time.strftime('%Y:%m:%d:%H:%M:%S.%f')
    return sim_start_time + sim_time


def create_num_charger_list(layouts_list, shared_charger_sets):
    """
    Create a list of number of chargers for each layout
    :param layouts_list:
    :param shared_charger_sets:
    :return:
    """
    return [len(shared_charger_sets) for _ in layouts_list]


def is_shared_charger(shared_charger_sets) -> bool:
    """
    Check if the vertiport has shared chargers
    :param shared_charger_sets:
    :return:
    """
    return shared_charger_sets is not None


def check_if_dataframe_is_empty(df):
    """
    Check if a dataframe is empty.
    :param df:
    :return:
    """
    return df.empty


def map_nested_dict_keys_to_values_from_list(dictionary, list_of_keys, last_key):
    """
    Map nested dictionary keys to values from a list input.
    :param dictionary:
    :param list_of_keys:
    :param last_key:
    :return:
    """
    return [dictionary[key][last_key] for key in list_of_keys]


def extract_dict_values(d):
    """
    Extracts all values from a nested dictionary with any depth.
    """
    values_list = []

    for key, value in d.items():
        if isinstance(value, dict):
            values_list.extend(extract_dict_values(value))
        else:
            values_list.append(value)

    return values_list


def nested_dict_all_values(d, depth):
    """
    Usage:
        list(nested_dict_values({1: {2: 3, 4: 5}}, 2))
        list(nested_dict_values(vertipad_PARK,2))
    """
    if depth == 1:
        for i in d.values():
            yield i
    else:
        for v in d.values():
            if isinstance(v, dict):
                for i in nested_dict_all_values(v, depth - 1):
                    yield i


def nested_dict_all_keys(d, depth):
    """
    Usage:
        list(nested_dict_keys({1: {2: 3, 4: 5}}, 2))
        list(nested_dict_keys(vertipad_PARK,2))
    """
    if depth == 1:
        for i in d.keys():
            yield i
    else:
        for v in d.values():
            if isinstance(v, dict):
                for i in nested_dict_all_keys(v, depth - 1):
                    yield i


def nested_dict_filter_keys(d, depth, key):
    """
    Usage:
        list(nested_dict_keys({1: {2: 3, 4: 5}}, 2))
        list(nested_dict_keys(vertipad_PARK,2))
    """
    if depth == 1:
        for i in d.keys():
            if i == key:
                yield d[key]
    else:
        for v in d.values():
            if isinstance(v, dict):
                for i in nested_dict_filter_keys(v, depth - 1, key):
                    yield i


def filter_nested_dict_by_given_keys(d: Dict, keys: List):
    """
    Filters a nested dictionary by given keys.
    """
    return {k: d[k] for k in keys if k in d}    


def get_str_before_first_occurrence_of_char(string, char):
    """
    Get string before first occurrence of a character.
    :param string:
    :param char:
    :return:
    """
    return string.split(char)[0]


def get_passenger_ids_from_passenger_list(passenger_list):
    """
    Get passenger ids from a passenger list.
    :param passenger_list:
    :return:
    """
    return [passenger.passenger_id for passenger in passenger_list]

def get_passenger_id_str_from_passenger_list(passenger_list):
    """
    Get passenger ids from a passenger list.
    :param passenger_list:
    :return:
    """
    return [f'passenger_{passenger.passenger_id}' for passenger in passenger_list]


def calculate_passenger_consolidation_time(departing_passengers):
    """
    Find the consolidation time by substracting the waiting room arrival time of the last passenger from the first passenger
    """
    return departing_passengers[-1].waiting_room_arrival_time - departing_passengers[0].waiting_room_arrival_time


def remove_larger_keys(dic1, dic2):
    # Deletes all the keys in dic1 that are greater than the highest key of dic2
    for key in list(dic1.keys()):
        if key > max(list(dic2.keys())):
            del dic1[key]
    return dic1
    

def get_values_from_nested_dict(dictionary, key):
    """
    Get values from a nested dictionary given second level key.
    :param dictionary:
    :param key:
    :return:
    """
    return [v[key] for v in dictionary.values()]


def get_length_of_longest_list(lst):
    return len(max(lst, key=len))


def get_length_of_shortest_list(lst):
    return len(min(lst, key=len))


def is_sqlite3(filename):
    from os.path import isfile, getsize

    if not isfile(filename):
        return False
    if getsize(filename) < 100: # SQLite database file header is 100 bytes
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[:16] == "SQLite format 3\x00"


def find_first_int(string):
    """
    Find the first integer in a string.
    """
    return int(re.findall(r'\d+', string)[0])


def find_second_int(string):
    """
    Find the second integer in a string.
    """
    return int(re.findall(r'\d+', string)[1])


def get_all_aircraft_models(network_and_demand: dict) -> set:
    return list(set(
        model 
        for vertiport_data in network_and_demand['vertiports'].values()
            for model in vertiport_data['aircraft_arrival_process']
            ['aircraft_fleet_specs'].keys()
            ))


def get_dict_vals_as_list(llist, key) -> List:
    """
    Returns a list of values from a list of dictionaries given a key.

    :param llist:
    :param key:
    """
    return [d[key] for d in llist]


def check_whether_node_exists(node_dict: Dict, node_id: Any):
    """
    Check whether a node exists in a dictionary of nodes.

    :param node_dict:
    :param node_id:
    """
    return node_id in node_dict.keys()


def check_if_none_exists(column: pd.Series) -> bool:
    """
    Check whether a column in a dataframe contains None values.

    :param df:
    :param column_name:
    """
    return column.isnull().values.any()


def check_if_all_none(column: pd.Series) -> bool:
    """
    Check whether a column in a dataframe contains only None values.

    :param df:
    :param column_name:
    """
    return column.isnull().values.all()

def convert_to_list(x: Any):
    """
    Convert an object to a list.
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, str):
        return [x]
    else:
        raise TypeError(
            f'Object is not a list, tuple or numpy array. The object is of type: {type(x)}. It is not supported.'
        )


def careful_round(value, decimal_places=2):
    """
    Round a value to 2 decimal places.
    """
    return None if value is None else round(value, decimal_places)

def careful_conversion(value):
    """
    Convert a value to float if it is not None.
    """
    if value is None:
        return None
    elif isinstance(value, str):
        return float(value)
    else:
        return value


def create_dict_for_fixes(airspace_nodes: Dict) -> Dict:
    """
    Create a dictionary for fixes.
    """
    fix_nodes = {'approach_fix_node': [], 'departure_fix_node': []}
    for key, value_list in airspace_nodes.items():
        for value in value_list:
            if 'approach' in value.lower():
                fix_nodes['approach_fix_node'].append(value)
            elif 'departure' in value.lower():
                fix_nodes['departure_fix_node'].append(value)
    return fix_nodes


def check_not_none(value):
    """
    Check whether a value is None.
    """
    return value is not None

def create_save_path(original_path, list_of_keys):
    # Create the new filename using the keys from the dictionary
    new_filename = f"network_{'_'.join(list_of_keys)}.csv"

    # Create the new path by joining the directory and the new filename
    return f"{original_path}/{new_filename}"

def check_magnitude_order(value1, value2, max_val=100):
    """
    Check whether the magnitude of value1 is greater than the magnitude of value2.
    """
    if value1 >= value2:
        raise ValueError(
            f'The magnitude of value1 ({value1}) is greater than or equal to the magnitude of value2 ({value2}).'
        )
    return value2 <= max_val
    
def miliseconds_to_hms(milliseconds):
    seconds = milliseconds // 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02d:%02d:%02d" % (hours, minutes, seconds)

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02d:%02d:%02d" % (hours, minutes, seconds)

import time
import string
def get_random_id():
    # Set simulation id
    random.seed(round(time.time()))
    return ''.join(random.choices(string.ascii_uppercase, k=8))

import uuid
def get_random_process_id():
    return str(uuid.uuid4())[:5]


def duplicate_str(string, n=5):
    """
    Duplicate a string n times.
    """
    return str(string) * n

def roundup(value, base=5):
    """
    Round up a value to the nearest base.
    """
    return math.ceil(value / base) * base

def roundup_to_five_minutes(ms):
    # 300,000 milliseconds is 5 minutes
    return ((ms + 299999) // 300000) * 300000


def lower_str(string: str) -> Optional[str]:
    """
    Convert a string to lowercase.
    """
    if string is None:
        return None
    elif isinstance(string, str):
        return string.lower()
    else:
        raise TypeError(
            f'Object is not a string. The object is of type: {type(string)}. It is not supported.'
        )
    
def get_list_elements_from_enum(enum_object: Enum):
    """
    Get a list of elements from an Enum.
    """
    return [object.name for object in enum_object]


def expected_pax_arrivals(arrival_df, demand_type, current_time, look_ahead_time, origin, destination):
    """
    Get the expected arrivals for the next x minutes.

    Parameters
    ----------
    arrival_df : pd.DataFrame
        Dataframe containing the arrival process for the demand type.
    demand_type : str
        Demand type. Can be 'rate_based' or 'scheduled'.
    current_time : int
        Current time in miliseconds.
    look_ahead_time : int
        Look ahead time in miliseconds.
    origin : str
        Origin vertiport id.
    destination : str
        Destination vertiport id.
    """
    if demand_type == 'rate_based':
        # Find the current time interval
        interval = arrival_df[arrival_df['time_sec'] <= ms_to_sec(current_time)].iloc[-1]

        route = f"{origin}_{destination}"

        if route in arrival_df.columns:
            lambda_rate = interval[route]
            expected_arrivals = round(ms_to_hr(lambda_rate * look_ahead_time))  # Convert to hours
            return expected_arrivals
        else:
            return "Route not found"
    elif demand_type == 'scheduled' or 'autoregressive':
        # Filter the data for the specified origin and destination
        filtered_data = arrival_df[(arrival_df['origin_vertiport_id'] == origin) & (arrival_df['destination_vertiport_id'] == destination)]

        # Filter the data to only include arrivals within the specified window
        time_window_data = filtered_data[(filtered_data['passenger_arrival_time'] >= current_time - look_ahead_time) & 
                                        (filtered_data['passenger_arrival_time'] < current_time)]

        # Estimate the expected arrivals in the next 10 minutes
        return len(time_window_data)

def expected_pax_arrivals_for_route(arrival_rate_df, current_time, look_ahead_time, route):
    """
    Get the expected arrivals for the next x minutes.

    Parameters
    ----------
    arrival_rate_df : pd.DataFrame
        Dataframe containing the arrival rates for each hour.
    current_time : int
        Current time in miliseconds.
    look_ahead_time : int
        Look ahead time in miliseconds.
    route : str
        Route id.
    """

    # Find the current time interval
    interval = arrival_rate_df[arrival_rate_df['time_sec'] <= ms_to_sec(current_time)].iloc[-1]

    if route in arrival_rate_df.columns:
        lambda_rate = interval[route]
        expected_arrivals = round(ms_to_hr(lambda_rate * look_ahead_time))  # Convert to hours
        return expected_arrivals
    else:
        return "Route not found"


import simpy
def get_stopping_events(env: simpy.Environment, stopping_events: List, aircraft_count: int, vertiport_ids: int, pax_count: int):
    """
    Get stopping events.
    """
    # return {event_name: env.event() for event_name in stopping_events}
    events = {}
    for event_name in stopping_events:
        if event_name in [
            "aircraft_faf_arrival_event",
            "aircraft_departure_fix_departure_event",
        ]:
            for vertiport_id in vertiport_ids:
                key = f"{event_name}_{vertiport_id}"
                events[key] = env.event()
        elif event_name in [
            "aircraft_parking_pad_arrival_event",
            "charging_start_event",
            "charging_end_event",
            "aircraft_departure_event"
        ]:
            for aircraft_id in range(aircraft_count):
                key = f"{event_name}_{aircraft_id}"
                events[key] = env.event()
        elif event_name in [
            "passenger_arrival_event"
        ]:
            for pax in range(pax_count):
                key = f"{event_name}_{pax}"
                events[key] = env.event()
    return events


def get_truncation_events(env: simpy.Environment, truncation_events: List, aircraft_count: int):
    """
    Get truncation events.
    """
    # return {event_name: env.event() for event_name in truncation_events}
    events = {}
    for event_name in truncation_events:
        for aircraft_id in range(aircraft_count):
            key = f"{event_name}_{aircraft_id}"
            events[key] = env.event()
    return events


def create_action_enum(vertiport_ids):
    class Action(Enum):
        # Enum members will be created dynamically
        pass

    # Dynamically add actions for flying to each vertiport
    for idx, vertiport_id in enumerate(vertiport_ids):
        Action._value2member_map_[idx] = vertiport_id

    # Add actions for charging and doing nothing
    Action._value2member_map_[len(vertiport_ids)] = 'CHARGE'
    Action._value2member_map_[len(vertiport_ids) + 1] = 'DO_NOTHING'

    return Action


def convert_action_values_to_enum(actions, action_enum: Enum):
    converted_actions = []
    for action_value in actions:
        try:
            _action_enum = action_enum(action_value)
            converted_actions.append(_action_enum)
        except ValueError:
            print(f"No enum member with value: {action_value}")
    return converted_actions


def convert_action_list_to_enum(action_list: List, action_enum: Enum):
    """
    Convert a list of actions to an enum.
    """
    return [action_enum(action) for action in action_list]


def create_action_dict(vertiport_ids):
    action_dict = {idx: vertiport_id for idx, vertiport_id in enumerate(vertiport_ids)}
    action_dict[len(vertiport_ids)] = 'CHARGE'
    action_dict[len(vertiport_ids) + 1] = 'DO_NOTHING'
    return action_dict


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def current_and_lookahead_pax_count(vertiport: object):
    """
    Get the current and lookahead pax count for the given vertiport.
    """
    return vertiport.get_total_waiting_passenger_count() + \
            vertiport.get_total_expected_pax_arrival_count()


def get_inflow_demand_to_vertiport(destination_vertiport_id: str, system_manager: object):
    """
    Get the inflow demand to a vertiport.
    """
    inflow_demand = 0
    for vertiport_id, vertiport in system_manager.vertiports.items():
        if vertiport_id != destination_vertiport_id:
            inflow_demand += vertiport.get_waiting_passenger_count()[destination_vertiport_id]
    return inflow_demand


def get_waiting_passenger_ids(sim_setup: object, origin_vertiport_id: str, destination_vertiport_id: str, exclude_pax_ids: List = []):
    """
    Returns the ids of the waiting passengers up to aircraft capacity
    """
    return sim_setup.vertiports[origin_vertiport_id].get_waiting_passenger_ids(destination=destination_vertiport_id, exclude_pax_ids=exclude_pax_ids)


def get_total_waiting_passengers_at_vertiport(sim_setup: object, vertiport_id: str):
    """
    Get the number of waiting passengers at the given vertiport.
    """
    return sim_setup.vertiports[vertiport_id].get_total_waiting_passenger_count()


def write_to_db(db_path, table_name, dic):
    """
    Write a dictionary to a database.

    :param db_path: Path to the SQLite database file.
    :param dic: Dictionary containing data to write.
    :param table_name: Name of the table to write to.    
    """
    import sqlite3
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    # Create a table with REAL data type for all columns
    columns = ', '.join([f"{key} REAL" for key in dic.keys()])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")    
    # Insert a row of data
    placeholders = ', '.join(['?'] * len(dic))
    cursor.execute(f"INSERT INTO {table_name} ({', '.join(dic.keys())}) VALUES ({placeholders})", list(dic.values()))
    # Save (commit) the changes
    conn.commit()
    # Close the connection
    conn.close()


def add_and_update_db_from_dict(db_path, table_name, new_column_name, new_column_type, data_dict):
    """
    Add a new column to an existing table in a SQLite database and update its values based on a dictionary.

    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table to modify.
    :param new_column_name: Name of the new column to add.
    :param new_column_type: Data type of the new column.
    :param data_dict: Dictionary with passenger_id as keys and the new column's values.
    """
    import sqlite3
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()

    # Add a new column to the table
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {new_column_name} {new_column_type}")

    # Prepare data for batch update
    update_data = [(value, key) for key, value in data_dict.items()]

    # Use executemany for more efficient batch updates
    cursor.executemany(f"UPDATE {table_name} SET {new_column_name} = ? WHERE passenger_id = ?", update_data)

    # Save (commit) the changes
    conn.commit()
    # Close the connection
    conn.close()




# def create_aircraft_events(env: simpy.Environment, aircraft_count: int):
#     """
#     Create a dedicated parking pad arrival event for each aircraft.
#     """
#     return {f"aircraft_{aircraft_id}_parking_pad_arrival_event": env.event() for aircraft_id in range(aircraft_count)}



# def nested_dict_filter_values(d, depth, value):
#     """
#     Usage:
#         list(nested_dict_values({1: {2: 3, 4: 5}}, 2))
#         list(nested_dict_values(vertipad_PARK,2))
#     """
#     if depth == 1:
#         for i in d.values():
#             if d[i] == value:
#                 yield i
#
#     else:
#         for v in d.values():
#             if isinstance(v, dict):
#                 for i in nested_dict_filter_values(v, depth - 1, value):
#                     yield i

# def inverse_cumcum(cumsum_array: np.ndarray) -> np.ndarray:
#     """
#     Inverse of cumulative sum array.
#     :param cumsum_array:
#     :return:
#     """
#     return np.diff(cumsum_array, prepend=0)
