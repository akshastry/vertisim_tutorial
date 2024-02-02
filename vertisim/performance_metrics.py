import numpy as np
import pandas as pd
from collections import defaultdict
from .utils.units import ms_to_min, sec_to_min, hr_to_ms
from .utils.helpers import nested_dict_all_values, remove_larger_keys, get_values_from_nested_dict, \
    get_length_of_longest_list, get_length_of_shortest_list, add_and_update_db_from_dict
# import matplotlib.pyplot as plt
from typing import List
import os


class PerformanceMetrics:

    def __init__(self, 
                sim_start_timestamp, 
                aircraft_params,
                only_aircraft_simulation, 
                is_network_simulation,
                performance_metric_trackers,
                vertiport_ids,
                flight_directions,
                sim_id, 
                sim_end_timestamp,
                sim_params,
                output_params,
                client_server_mode,
                verbose=False):
        self.sim_start_timestamp = sim_start_timestamp
        self.aircraft_params = aircraft_params
        self.only_aircraft_simulation = only_aircraft_simulation
        self.is_network_simulation = is_network_simulation
        self.client_server_mode = client_server_mode
        self.sim_params = sim_params
        self.output_params = output_params
        self.performance_metric_trackers = performance_metric_trackers
        self.passenger_consolidation_time = performance_metric_trackers['passenger_consolidation_time']
        self.passenger_departure_queue_count = performance_metric_trackers['passenger_departure_queue_count']
        self.aircraft_arrival_queue_count = performance_metric_trackers['aircraft_arrival_queue_count']
        self.aircraft_departure_queue_count = performance_metric_trackers['aircraft_departure_queue_count']
        self.passenger_waiting_time = performance_metric_trackers['passenger_waiting_time']
        self.passenger_transfer_time = performance_metric_trackers['passenger_transfer_time']
        self.passenger_departure_queue_waiting_time = performance_metric_trackers['passenger_departure_queue_waiting_time']
        self.aircraft_holding_time = performance_metric_trackers['aircraft_holding_time']
        self.aircraft_charging_times = performance_metric_trackers['aircraft_charging_times']
        self.aircraft_taxi_times = performance_metric_trackers['aircraft_taxi_times']
        self.flight_schedule = performance_metric_trackers['flight_schedule']
        self.fato_usage_tracker = performance_metric_trackers['fato_usage_tracker']
        self.aircraft_idle_time_tracker = performance_metric_trackers['aircraft_idle_time_tracker']
        self.flight_and_time_tracker = performance_metric_trackers['flight_and_time_tracker']
        self.flight_count_tracker = performance_metric_trackers['flight_count_tracker']
        self.flight_duration_tracker = performance_metric_trackers['flight_duration_tracker']
        self.load_factor_tracker = performance_metric_trackers['load_factor_tracker']
        self.passenger_count_tracker = performance_metric_trackers['passenger_count_tracker']
        self.passenger_time_tracker = performance_metric_trackers['passenger_time_tracker']
        self.aircraft_energy_consumption_tracker = performance_metric_trackers['aircraft_energy_consumption_tracker']
        self.passenger_trip_times = performance_metric_trackers['passenger_trip_time_tracker']
        self.repositioning_counter = performance_metric_trackers['repositioning_counter']
        self.total_charge_time_counter = performance_metric_trackers['total_charge_time_counter']
        self.output_folder_path = self.get_save_path()
        self.vertiport_ids = vertiport_ids
        self.flight_directions = flight_directions
        self.sim_id = sim_id
        self.sim_end_timestamp = sim_end_timestamp
        self.hour = 60 * 60 * 1000
        self.performance_metrics = defaultdict(lambda: defaultdict(dict))
        # TODO: Remove this. Temporary fix
        # path = '../../output/performance_metrics'
        # if not os.path.exists(path):
        #     os.makedirs(path)

        if not self.only_aircraft_simulation:
            self.calculate_passenger_consolidation_time_stats()
            self.calculate_passenger_waiting_time_stats()
            self.calculate_passenger_transfer_time_stats()
            self.calculate_passenger_departure_queue_count_stats()
            self.calculate_passenger_trip_time_stats()
            self.compute_total_passenger_throughput()
            self.compute_average_passenger_throughput()
            self.compute_load_factor_stats()
            self.compute_empty_flights()
            self.compute_aircraft_idle_time_stats()
            self.calculate_passenger_departure_queue_waiting_times_stats()
        self.calculate_aircraft_arrival_queue_count_stats()
        self.calculate_aircraft_departure_queue_count_stats()
        self.compute_average_and_total_aircraft_throughput()
        if self.sim_params['training_data_collection']:
            self.save_waiting_times_to_db()
        self.compute_fato_utilization()
        self.calculate_aircraft_holding_times_stats()
        self.compute_flight_duration_stats()
        self.compute_energy_consumption_stats()
        self.compute_charge_time_stats()
        self.compute_average_taxi_time()
        self.compute_turnaround_duration()
        self.compute_tlof_reservation_duration()

        if verbose and not self.client_server_mode:
            self.print_verbose_performance_metrics()
        elif not self.client_server_mode:
            self.print_performance_metrics()

        # TODO: Remove all save to csv methods from this file. save_output should be the only method that saves to csv
        # df = pd.DataFrame(self.flight_and_time_tracker)
        # df.to_csv('../../output/performance_metrics/flight_and_time_tracker.csv')  
        # 
    def get_save_path(self):
        path = f'{self.output_params["output_folder_path"]}/{self.output_params["performance_metrics_output_file_name"]}'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
    
    def save_waiting_times_to_db(self):
        df = pd.DataFrame(self.passenger_waiting_time)
        # Merge the columns into one. This requires all columns to be NaN except one of them
        df = df.bfill(axis=1).iloc[:, 0]
        # Sorting by index
        df.sort_index(inplace=True)
        # Convert the unit of time from ms to min
        df = df.apply(ms_to_min)
        # Rename the column
        df = df.rename('waiting_time_min')
        # Convert it to a dict
        # df = df.reset_index().rename(columns={'index': 'passenger_id'})
        df = df.to_dict()
        # Write to the database
        add_and_update_db_from_dict(db_path=self.output_params['state_trajectory_db_path'],
                                    table_name=self.output_params['state_trajectory_db_tablename'],
                                    new_column_name='waiting_time_min',
                                    new_column_type='REAL',
                                    data_dict=df)

        # df.to_csv(f'{self.output_folder_path}/passenger_waiting_time.csv')   



    def calculate_passenger_consolidation_time_stats(self): 
        for vertiport_id, data in self.passenger_consolidation_time.items():
            if consolidation_times := [
                details['consolidation_time'] for details in data.values()
            ]:
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['mean'] = ms_to_min(np.mean(consolidation_times))
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['median'] = ms_to_min(np.median(consolidation_times))
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['std'] = ms_to_min(np.std(consolidation_times))
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['max'] = ms_to_min(np.max(consolidation_times))
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['min'] = ms_to_min(np.min(consolidation_times))
            else:
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['mean'] = -1
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['median'] = -1
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['std'] = -1
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['max'] = -1
                self.performance_metrics[vertiport_id]['passenger_consolidation_time']['min'] = -1

    def calculate_passenger_waiting_time_stats(self):
        # Creates a dataframe for passenger waiting time statistics including mean, median, standard deviation,
        for vertiport_id, data in self.passenger_waiting_time.items():
            waiting_times = np.array(list(data.values()))
            if waiting_times.size > 0:
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['mean'] = ms_to_min(np.mean(waiting_times))
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['median'] = ms_to_min(np.median(waiting_times))
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['std'] = ms_to_min(np.std(waiting_times))
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['max'] = ms_to_min(np.max(waiting_times))
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['min'] = ms_to_min(np.min(waiting_times))
            else:
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['mean'] = -1
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['median'] = -1
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['std'] = -1
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['max'] = -1
                self.performance_metrics[vertiport_id]['passenger_waiting_time']['min'] = -1

    def calculate_passenger_transfer_time_stats(self):
        # Creates a dataframe for passenger transfer time statistics including mean, median, standard deviation,
        for vertiport_id, data in self.passenger_transfer_time.items():
            transfer_times = np.array(list(data.values()))

            self.performance_metrics[vertiport_id]['passenger_transfer_time']['mean'] = ms_to_min(np.mean(transfer_times))
            self.performance_metrics[vertiport_id]['passenger_transfer_time']['median'] = ms_to_min(np.median(transfer_times))
            self.performance_metrics[vertiport_id]['passenger_transfer_time']['std'] = ms_to_min(np.std(transfer_times))
            self.performance_metrics[vertiport_id]['passenger_transfer_time']['max'] = ms_to_min(np.max(transfer_times))
            self.performance_metrics[vertiport_id]['passenger_transfer_time']['min'] = ms_to_min(np.min(transfer_times))

    def calculate_passenger_trip_time_stats(self):
        # df = pd.DataFrame(self.passenger_trip_times)
        # df.to_csv('../../output/performance_metrics/passenger_trip_times.csv')           
        # Creates a dataframe for passenger trip time statistics including mean, median, standard deviation,
        for flight_direction, data in self.passenger_trip_times.items():
            trip_times = np.array(list(data.values()))

            self.performance_metrics[flight_direction]['passenger_trip_time']['mean'] = ms_to_min(np.mean(trip_times))
            self.performance_metrics[flight_direction]['passenger_trip_time']['median'] = ms_to_min(np.median(trip_times))
            self.performance_metrics[flight_direction]['passenger_trip_time']['std'] = ms_to_min(np.std(trip_times))
            self.performance_metrics[flight_direction]['passenger_trip_time']['max'] = ms_to_min(np.max(trip_times))
            self.performance_metrics[flight_direction]['passenger_trip_time']['min'] = ms_to_min(np.min(trip_times))

    def calculate_passenger_departure_queue_count_stats(self):
        # Creates a dataframe for passenger departure queue count statistics including mean, median, standard deviation,
        for vertiport_id, data in self.passenger_departure_queue_count.items():
            passenger_departure_queue_count = np.array(list(data.values()))
            self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['mean'] = np.mean(passenger_departure_queue_count)
            self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['median'] = np.median(passenger_departure_queue_count)
            self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['std'] = np.std(passenger_departure_queue_count)
            self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['max'] = np.max(passenger_departure_queue_count)
            self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['min'] = np.min(passenger_departure_queue_count)

    def calculate_passenger_departure_queue_waiting_times_stats(self):
        # Creates a dataframe for passenger departure queue waiting times statistics including mean, median, standard deviation,
        # max and min
        for vertiport_id, data in self.passenger_departure_queue_waiting_time.items():
            waiting_times = np.array(list(data.values()))
            self.performance_metrics[vertiport_id][
                'passenger_departure_queue_waiting_times']['mean'] = ms_to_min(np.mean(waiting_times))
            self.performance_metrics[vertiport_id][
                'passenger_departure_queue_waiting_times']['median'] = ms_to_min(np.median(waiting_times))
            self.performance_metrics[vertiport_id][
                'passenger_departure_queue_waiting_times']['std'] = ms_to_min(np.std(waiting_times))
            self.performance_metrics[vertiport_id][
                'passenger_departure_queue_waiting_times']['max'] = ms_to_min(np.max(waiting_times))
            self.performance_metrics[vertiport_id][
                'passenger_departure_queue_waiting_times']['min'] = ms_to_min(np.min(waiting_times))

    def calculate_aircraft_arrival_queue_count_stats(self):
        # Creates a dataframe for aircraft arrival queue count statistics including mean, median, standard deviation,
        for vertiport_id, data in self.aircraft_arrival_queue_count.items():
            aircraft_arrival_queue_count = np.array(list(data.values()))
            self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['mean'] = np.mean(aircraft_arrival_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['median'] = np.median(aircraft_arrival_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['std'] = np.std(aircraft_arrival_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['max'] = np.max(aircraft_arrival_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['min'] = np.min(aircraft_arrival_queue_count)

    def calculate_aircraft_departure_queue_count_stats(self):
        # Creates a dataframe for aircraft departure queue count statistics including mean, median, standard deviation,
        for vertiport_id, data in self.aircraft_departure_queue_count.items():
            aircraft_departure_queue_count = np.array(list(data.values()))
            self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['mean'] = np.mean(aircraft_departure_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['median'] = np.median(aircraft_departure_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['std'] = np.std(aircraft_departure_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['max'] = np.max(aircraft_departure_queue_count)
            self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['min'] = np.min(aircraft_departure_queue_count)

    def compute_flight_duration_stats(self):
        # Creates a dictionary entry for flight time statistics including mean, median, standard deviation,
        for flight_direction, data in self.flight_duration_tracker.items():
            flight_durations = np.array(list(data.values()))
            self.performance_metrics[flight_direction]['flight_duration']['mean'] = ms_to_min(np.mean(flight_durations))
            self.performance_metrics[flight_direction]['flight_duration']['median'] = ms_to_min(np.median(flight_durations))
            self.performance_metrics[flight_direction]['flight_duration']['std'] = ms_to_min(np.std(flight_durations))
            self.performance_metrics[flight_direction]['flight_duration']['max'] = ms_to_min(np.max(flight_durations))
            self.performance_metrics[flight_direction]['flight_duration']['min'] = ms_to_min(np.min(flight_durations))

    def compute_energy_consumption_stats(self):
        for flight_direction, data in self.aircraft_energy_consumption_tracker.items():
            energy_consumptions = np.array(list(data.values()))
            self.performance_metrics[flight_direction]['energy_consumption']['mean'] = np.mean(energy_consumptions)
            self.performance_metrics[flight_direction]['energy_consumption']['median'] = np.median(energy_consumptions)
            self.performance_metrics[flight_direction]['energy_consumption']['std'] = np.std(energy_consumptions)
            self.performance_metrics[flight_direction]['energy_consumption']['max'] = np.max(energy_consumptions)
            self.performance_metrics[flight_direction]['energy_consumption']['min'] = np.min(energy_consumptions)
            self.performance_metrics[flight_direction]['energy_consumption']['total'] = np.sum(energy_consumptions)

    def calculate_aircraft_holding_times_stats(self):
        df = pd.DataFrame(self.aircraft_holding_time)
        for vertiport_id, data in self.aircraft_holding_time.items():
            waiting_times = np.array(list(data.values()))
            self.performance_metrics[vertiport_id]['aircraft_holding_times']['mean'] = ms_to_min(np.mean(waiting_times))
            self.performance_metrics[vertiport_id]['aircraft_holding_times']['median'] = ms_to_min(np.median(waiting_times))
            self.performance_metrics[vertiport_id]['aircraft_holding_times']['std'] = ms_to_min(np.std(waiting_times))
            self.performance_metrics[vertiport_id]['aircraft_holding_times']['max'] = ms_to_min(np.max(waiting_times))
            self.performance_metrics[vertiport_id]['aircraft_holding_times']['min'] = ms_to_min(np.min(waiting_times))

    def compute_load_factor_stats(self):
        for vertiport_id, load_factors in self.load_factor_tracker.items():
            self.performance_metrics[vertiport_id]['load_factor']['max'] = np.max(load_factors)
            self.performance_metrics[vertiport_id]['load_factor']['min'] = np.min(load_factors)
            self.performance_metrics[vertiport_id]['load_factor']['mean'] = np.mean(load_factors)

    def compute_aircraft_idle_time_stats(self):
        df = pd.DataFrame(self.aircraft_idle_time_tracker)
        for vertiport_id, data in self.aircraft_idle_time_tracker.items():
            idle_times = list(data.values())
            self.performance_metrics[vertiport_id]['aircraft_idle_time']['max'] = ms_to_min(np.max(idle_times))
            self.performance_metrics[vertiport_id]['aircraft_idle_time']['min'] = ms_to_min(np.min(idle_times))
            self.performance_metrics[vertiport_id]['aircraft_idle_time']['mean'] = ms_to_min(np.mean(idle_times))
            self.performance_metrics[vertiport_id]['aircraft_idle_time']['median'] = ms_to_min(np.median(idle_times))
            self.performance_metrics[vertiport_id]['aircraft_idle_time']['std'] = ms_to_min(np.std(idle_times))

    def compute_charge_time_stats(self):
        # Computes the average charge time
        if self.aircraft_charging_times:
            for vertiport_id in self.vertiport_ids:
                if charge_times := list(self.aircraft_charging_times['vertiport_id'].values()):
                    self.performance_metrics[vertiport_id]['charge_time']['mean'] = ms_to_min(np.mean(charge_times))
                    self.performance_metrics[vertiport_id]['charge_time']['median'] = ms_to_min(np.median(charge_times))
                    self.performance_metrics[vertiport_id]['charge_time']['std'] = ms_to_min(np.std(charge_times))
                    self.performance_metrics[vertiport_id]['charge_time']['max'] = ms_to_min(np.max(charge_times))
                    self.performance_metrics[vertiport_id]['charge_time']['min'] = ms_to_min(np.min(charge_times))
                else:
                    self.performance_metrics[vertiport_id]['charge_time']['mean'] = 0
        else:
            for vertiport_id in self.vertiport_ids:
                self.performance_metrics[vertiport_id]['charge_time']['mean'] = 0

    def compute_average_taxi_time(self):
        # Computes the average taxi time
        for vertiport_id, data in self.aircraft_taxi_times.items():
            taxi_times = list(data.values())
            self.performance_metrics[vertiport_id]['average_taxi_time'] = ms_to_min(np.mean(taxi_times))

    def compute_turnaround_duration(self):
        # Computes the average aircraft turnaround time
        for vertiport_id in self.vertiport_ids:
            self.performance_metrics[vertiport_id]['average_aircraft_turnaround_time'] = round(
                2*self.performance_metrics[vertiport_id]['average_taxi_time'] + \
                self.performance_metrics[vertiport_id]['charge_time']['mean'] + \
                sec_to_min(self.aircraft_params['time_pre_charging_processes']) + \
                sec_to_min(self.aircraft_params['time_post_charging_processes']) + \
                sec_to_min(self.aircraft_params['time_charging_plug_disconnection']), 2)
    
    def compute_tlof_reservation_duration(self):
        for vertiport_id in self.vertiport_ids:
            tlof_time = self.aircraft_params['time_descend_transition'] + \
                self.aircraft_params['time_hover_descend'] + \
                self.aircraft_params['time_rotor_spin_down'] + \
                self.aircraft_params['time_post_landing_safety_checks'] + \
                self.aircraft_params['time_tug_connection'] + \
                self.aircraft_params['time_pre_take_off_check_list'] + \
                self.aircraft_params['time_rotor_spin_up'] + \
                self.aircraft_params['time_hover_climb'] + \
                self.aircraft_params['time_climb_transition'] + \
                self.performance_metrics[vertiport_id]['average_taxi_time']
            tlof_time = tlof_time / 2
            self.performance_metrics[vertiport_id]['tlof_time'] = ms_to_min(tlof_time)

    def compute_total_passenger_throughput(self):
        # Computes the total passenger throughput
        for vertiport_id in self.vertiport_ids:
            self.performance_metrics[vertiport_id]['passenger_throughput'] = self.passenger_count_tracker[vertiport_id]
    
    def compute_average_passenger_throughput(self):
        # Computes the passenger throughput skipping the first 1 hour of the simulation
        total_sim_time = self.calculate_total_simulated_time()
        if self.is_network_simulation:
            for vertiport_id in self.vertiport_ids:
                self.performance_metrics[vertiport_id]['average_passenger_throughput'] = round(
                    self.passenger_count_tracker[vertiport_id] / total_sim_time, 2
                )
        else:
            first, second = self.get_time_interval(total_sim_time)
            interval_length = second - first
            for vertiport_id in self.vertiport_ids:
                total_passengers = sum(
                    value
                    for key, value in self.passenger_time_tracker[vertiport_id].items()
                    if self.sim_start_timestamp + self.hour * first
                    <= key
                    <= self.sim_start_timestamp + second * self.hour
                )
                self.performance_metrics[vertiport_id]['average_passenger_throughput'] = round(total_passengers / interval_length, 2)

    def compute_empty_flights(self):
        # Computes the number of empty flights
        for vertiport_id in self.vertiport_ids:
            self.performance_metrics[vertiport_id]['empty_flights'] = 0
            for _, value in self.passenger_time_tracker[vertiport_id].items():
                if value == 0:
                    self.performance_metrics[vertiport_id]['empty_flights'] += 1

    def get_time_interval(self, total_sim_time: int) -> List[int]:
        """
        Returns the time interval for the result indexing.
        :param total_sim_time: Total simulation time
        :return: Start of interval, end of interval
        """
        if total_sim_time < 4:
            return 0, 2
            # raise ValueError(f'Simulation time is less than 4 hours ({total_sim_time} hrs). Simulation ID: {self.sim_id}')
        elif 4 < total_sim_time <= 5:
            return 2, 4
        elif 5 < total_sim_time <= 6:
            return 2, 5
        elif 6 < total_sim_time <= 7:
            return 2, 5
        elif 7 < total_sim_time <= 8:
            return 3, 6
        elif 8 < total_sim_time <= 9:
            return 3, 7
        elif 9 < total_sim_time <= 10:
            return 3, 8
        elif 10 < total_sim_time <= 11:
            return 4, 9
        elif 11 < total_sim_time <= 12:
            return 4, 9
        elif 12 < total_sim_time <= 15:
            return 4, 10
        elif 15 < total_sim_time <= 20:
            return 5, 13
        else:
            return 6, 15

    def compute_average_and_total_aircraft_throughput(self):
        # Computes the passenger throughput
        total_sim_time = self.calculate_total_simulated_time()
        if self.is_network_simulation:
            for vertiport_id in self.vertiport_ids:
                total_aircraft = sum(value 
                                     for key, value 
                                     in self.flight_and_time_tracker[vertiport_id].items()
                                     )    
                self.performance_metrics[vertiport_id]['aircraft_throughput']['mean'] = round(total_aircraft / total_sim_time, 2)
                self.performance_metrics[vertiport_id]['aircraft_throughput']['total'] = self.flight_count_tracker[vertiport_id]
        else:
            first, second = self.get_time_interval(total_sim_time)
            interval_length = second - first
            for vertiport_id in self.vertiport_ids:
                total_aircraft = 0
                total_aircraft += sum(
                    value
                    for key, value in self.flight_and_time_tracker[vertiport_id].items()
                    if self.sim_start_timestamp + self.hour * first
                    <= key
                    <= self.sim_start_timestamp + second * self.hour
                )
                self.performance_metrics[vertiport_id]['aircraft_throughput']['mean'] = total_aircraft / interval_length
                self.performance_metrics[vertiport_id]['aircraft_throughput']['total'] = self.flight_count_tracker[vertiport_id]

    def compute_aircraft_throughput_for_each_hour(self):
        # Computes the aircraft throughput for each hour
        total_sim_time = self.calculate_total_simulated_time() 
        total_aircraft_at_each_hour = defaultdict(dict)        
        if self.is_network_simulation:       
            for vertiport_id in self.vertiport_ids:
                for time_interval in range(int(total_sim_time)):
                    total_aircraft_at_each_hour[vertiport_id][time_interval] = 0
                    for key, value in self.flight_and_time_tracker[vertiport_id].items():
                        if self.sim_start_timestamp + self.hour * time_interval <= key <= self.sim_start_timestamp + self.hour * (time_interval + 1):
                            total_aircraft_at_each_hour[vertiport_id][time_interval] += 1             
        else:
            first, second = self.get_time_interval(total_sim_time)
            for vertiport_id in self.vertiport_ids:
                for time_interval in range(first, second):
                    total_aircraft_at_each_hour[vertiport_id][time_interval] = 0
                    for key, value in self.flight_and_time_tracker[vertiport_id].items():
                        if self.sim_start_timestamp + self.hour * time_interval <= key <= self.sim_start_timestamp + self.hour * (time_interval + 1):
                            total_aircraft_at_each_hour[vertiport_id][time_interval] += 1
        return total_aircraft_at_each_hour

    def compute_fato_utilization(self):
        # Sums the difference between keys when the first value is 0 and the next value is 1.
        total_sim_time = self.calculate_total_simulated_time()
        for vertiport_id, data in self.fato_usage_tracker.items():
            time_fato_empty = 0
            occupancy_index = 0
            if self.is_network_simulation:
                keys = list(data.keys())
                values = list(data.values())
                for i in range(len(keys)):
                    # if values[i] == 0 and values[i + 1] == 1:
                    #     time_fato_empty += keys[i + 1] - keys[i]
                    if values[i] == 1:
                        time_fato_empty += keys[i] - keys[occupancy_index]
                        occupancy_index = i+1
            else:
                keys = list(data.keys())
                values = list(data.values())
                for i in range(len(keys)):
                    if keys[i] >= self.sim_start_timestamp + self.hour:
                        if values[i] == 1:
                            time_fato_empty += keys[i] - keys[occupancy_index]
                            occupancy_index = i+1

            fato_utilization = round((hr_to_ms(total_sim_time) - time_fato_empty) / hr_to_ms(total_sim_time) * 100, 2)
            self.performance_metrics[vertiport_id]['fato_utilization'] = fato_utilization

    def print_passenger_consolidation_time_stats(self):
        print("Passenger consolidation time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['passenger_consolidation_time']['mean'],
                'median': self.performance_metrics[vertiport_id]['passenger_consolidation_time']['median'],
                'std': self.performance_metrics[vertiport_id]['passenger_consolidation_time']['std'],
                'max': self.performance_metrics[vertiport_id]['passenger_consolidation_time']['max'],
                'min': self.performance_metrics[vertiport_id]['passenger_consolidation_time']['min']
            } for vertiport_id in self.vertiport_ids if self.performance_metrics[vertiport_id]['passenger_consolidation_time']
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_passenger_waiting_time_stats(self):
        print("Passenger waiting time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['passenger_waiting_time']['mean'],
                'median': self.performance_metrics[vertiport_id]['passenger_waiting_time']['median'],
                'std': self.performance_metrics[vertiport_id]['passenger_waiting_time']['std'],
                'max': self.performance_metrics[vertiport_id]['passenger_waiting_time']['max'],
                'min': self.performance_metrics[vertiport_id]['passenger_waiting_time']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_passenger_transfer_time_stats(self):
        print("Passenger transfer time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['passenger_transfer_time']['mean'],
                'median': self.performance_metrics[vertiport_id]['passenger_transfer_time']['median'],
                'std': self.performance_metrics[vertiport_id]['passenger_transfer_time']['std'],
                'max': self.performance_metrics[vertiport_id]['passenger_transfer_time']['max'],
                'min': self.performance_metrics[vertiport_id]['passenger_transfer_time']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_passenger_trip_time_stats(self):
        print("Passenger trip time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[flight_direction]['passenger_trip_time']['mean'],
                'median': self.performance_metrics[flight_direction]['passenger_trip_time']['median'],
                'std': self.performance_metrics[flight_direction]['passenger_trip_time']['std'],
                'max': self.performance_metrics[flight_direction]['passenger_trip_time']['max'],
                'min': self.performance_metrics[flight_direction]['passenger_trip_time']['min']
            } for flight_direction in self.flight_directions
        )
        df.index = self.flight_directions
        print(df)
        print("")

    def print_passenger_departure_queue_count_stats(self):
        print("Passenger departure queue count statistics:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['mean'],
                'median': self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['median'],
                'std': self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['std'],
                'max': self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['max'],
                'min': self.performance_metrics[vertiport_id]['passenger_departure_queue_count']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_aircraft_arrival_queue_count_stats(self):
        print("Aircraft arrival queue count statistics:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['mean'],
                'median': self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['median'],
                'std': self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['std'],
                'max': self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['max'],
                'min': self.performance_metrics[vertiport_id]['aircraft_arrival_queue_count']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_aircraft_holding_times_stats(self):
        print("\nAircraft holding time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['aircraft_holding_times']['mean'],
                'median': self.performance_metrics[vertiport_id]['aircraft_holding_times']['median'],
                'std': self.performance_metrics[vertiport_id]['aircraft_holding_times']['std'],
                'max': self.performance_metrics[vertiport_id]['aircraft_holding_times']['max'],
                'min': self.performance_metrics[vertiport_id]['aircraft_holding_times']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_aircraft_departure_queue_count_stats(self):
        print("Aircraft departure queue count statistics:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['mean'],
                'median': self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['median'],
                'std': self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['std'],
                'max': self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['max'],
                'min': self.performance_metrics[vertiport_id]['aircraft_departure_queue_count']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_aircraft_idle_time_stats(self):
        print(" ")
        print("Aircraft idle time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[vertiport_id]['aircraft_idle_time']['mean'],
                'median': self.performance_metrics[vertiport_id]['aircraft_idle_time']['median'],
                'std': self.performance_metrics[vertiport_id]['aircraft_idle_time']['std'],
                'max': self.performance_metrics[vertiport_id]['aircraft_idle_time']['max'],
                'min': self.performance_metrics[vertiport_id]['aircraft_idle_time']['min']
            } for vertiport_id in self.vertiport_ids
        )
        df.index = self.vertiport_ids
        print(df)
        print("")

    def print_flight_duration_stats(self):
        print(" ")
        print("Flight time statistics in minutes:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[flight_direction]['flight_duration']['mean'],
                'median': self.performance_metrics[flight_direction]['flight_duration']['median'],
                'std': self.performance_metrics[flight_direction]['flight_duration']['std'],
                'max': self.performance_metrics[flight_direction]['flight_duration']['max'],
                'min': self.performance_metrics[flight_direction]['flight_duration']['min']
            } for flight_direction in self.flight_duration_tracker.keys()
        )
        df.index = self.flight_duration_tracker.keys()
        print(df)
        print("")

    def print_energy_consumption_stats(self):
        print(" ")
        print("Energy consumption statistics in kWh:")
        print("-----------------------------")
        df = pd.DataFrame(
            {
                'mean': self.performance_metrics[flight_direction]['energy_consumption']['mean'],
                'median': self.performance_metrics[flight_direction]['energy_consumption']['median'],
                'std': self.performance_metrics[flight_direction]['energy_consumption']['std'],
                'max': self.performance_metrics[flight_direction]['energy_consumption']['max'],
                'min': self.performance_metrics[flight_direction]['energy_consumption']['min'],
                'total': self.performance_metrics[flight_direction]['energy_consumption']['total']
            } for flight_direction in self.aircraft_energy_consumption_tracker.keys()
        )
        df.index = self.aircraft_energy_consumption_tracker.keys()
        print(df)
        print("")

    def print_aircraft_throughput_stats_for_each_hr_verbose(self):
        print(" ")
        print("Aircraft throughput statistics for each hour:")
        print("-----------------------------")
        total_aircraft_at_each_hour = self.compute_aircraft_throughput_for_each_hour()
        for vertiport_id in self.vertiport_ids:
            total_aircraft = list(total_aircraft_at_each_hour[vertiport_id].values())
            print(f'Max number of departures during any 1-hr interval at {vertiport_id}: {round(np.max(total_aircraft), 2)}')
            print(f'Min number of departures during any 1-hr interval at {vertiport_id}: {round(np.min(total_aircraft), 2)}')
            print(f'Std of departures during 1 hr interval: {round(np.std(total_aircraft), 2)}')

    # def plot_aircraft_arrival_queue_count(self):
    #     # TODO: Fix this
    #     # Plots aircraft arrival queue count
    #     plt.plot(
    #         np.round((np.array(list(self.aircraft_arrival_queue_count.keys())) - self.sim_start_timestamp) / 60000),
    #         list(self.aircraft_arrival_queue_count.values()),
    #         alpha=0.4)
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Aircraft Arrival Queue Count')
    #     plt.title('Aircraft Arrival Queue Count Over Time')
    #     plt.show()

    def calculate_total_simulated_time(self):
        return round(self.sim_end_timestamp/self.hour, 2)
    
    def register_total_simulated_time(self):
        # Registers total simulated time
        self.performance_metrics['total_simulated_time'] = self.calculate_total_simulated_time()

    def print_total_simulated_time(self):
        print(f"{self.calculate_total_simulated_time()} hours simulated")

    def print_avg_load_factor(self):
        for vertiport_id in self.vertiport_ids:
            print(f"Average load factor for non-empty flights at {vertiport_id}: {round(self.performance_metrics[vertiport_id]['load_factor']['mean'], 2)}")

    def print_total_aircraft_throughput(self):
        for vertiport_id in self.vertiport_ids:
            print(f"Total aircraft throughput at {vertiport_id}: {self.performance_metrics[vertiport_id]['aircraft_throughput']['total']}")

    def print_average_aircraft_throughput(self):
        print(" ")
        for vertiport_id in self.vertiport_ids:
            print(f"Average aircraft throughput in 1 hour at {vertiport_id}: {round(self.performance_metrics[vertiport_id]['aircraft_throughput']['mean'], 2)}")

    def print_total_passenger_throughput(self):
        for vertiport_id in self.vertiport_ids:
            print(f"Total passenger throughput at {vertiport_id}: {self.performance_metrics[vertiport_id]['passenger_throughput']}")

    def print_average_passenger_throughput(self):
        for vertiport_id in self.vertiport_ids:
            print(f"Average passenger throughput in 1 hour at {vertiport_id}: {round(self.performance_metrics[vertiport_id]['average_passenger_throughput'], 2)}")

    def print_average_charge_time(self):
        for vertiport_id in self.vertiport_ids:
            print(f"Average charge time at {vertiport_id}: {self.performance_metrics[vertiport_id]['charge_time']['mean']}")
        print("Total charge time (min): ", round(self.total_charge_time_counter/1000/60, 2))

    def print_fato_utilization(self):
        print(" ")
        for vertiport_id in self.vertiport_ids:
            print(f"FATO utilization at {vertiport_id}: {self.performance_metrics[vertiport_id]['fato_utilization']}%")

    def print_total_empty_flights(self):
        print(" ")
        for vertiport_id in self.vertiport_ids:
            print(f"Total empty flights from {vertiport_id}: {self.performance_metrics[vertiport_id]['empty_flights']}")
        
        for vertiport_id in self.vertiport_ids:
            print(f"Total repositioning flights from {vertiport_id}: {self.repositioning_counter[vertiport_id]}")

    def print_verbose_performance_metrics(self):
        if not self.only_aircraft_simulation:
            self.print_passenger_consolidation_time_stats()
            self.print_passenger_waiting_time_stats()
            # self.print_passenger_transfer_time_stats()
            self.print_passenger_trip_time_stats()
            # self.print_passenger_departure_queue_count_stats()
            self.print_avg_load_factor()
            self.print_aircraft_idle_time_stats()
            self.print_total_passenger_throughput()
            self.print_average_passenger_throughput()
        # self.print_aircraft_arrival_queue_count_stats()
        # self.print_aircraft_departure_queue_count_stats()
        self.print_aircraft_holding_times_stats()
        self.print_average_charge_time()
        self.print_total_simulated_time()
        self.print_total_aircraft_throughput()
        self.print_flight_duration_stats()
        self.print_energy_consumption_stats()
        self.print_aircraft_throughput_stats_for_each_hr_verbose()
        self.print_average_aircraft_throughput()
        self.print_fato_utilization()
        self.print_total_empty_flights()

    def print_performance_metrics(self):
        if not self.only_aircraft_simulation:
            self.print_avg_load_factor()
            self.print_total_passenger_throughput()
            self.print_average_passenger_throughput()
        self.print_total_simulated_time()
        self.print_average_charge_time()
        self.print_total_aircraft_throughput()
        self.print_average_aircraft_throughput()
        self.print_fato_utilization()
        self.print_total_empty_flights()
        
