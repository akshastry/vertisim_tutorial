from loop_config import *
import itertools
from tqdm.contrib.itertools import product
import numpy as np
import pandas as pd


def run_vertisim():

    vertiport1 = ['SFO']
    vertiport2 = ['OAK'] #['MV', 'SJ', 'SANTACRUZ', 'STOCKTON']
    network_id = ['SFO_OAK'] #['SFO_MV', 'SFO_SJ', 'SFO_SANTACRUZ', 'SFO_STOCKTON']
    num_park = [7]
    fleet_size = [num_park[0]*2]
    min_reserve_soc = [30]
    target_soc_constant = [40]

    # min_reserve_soc = [30, 33, 36.5]
    # target_soc_constant = [40, 46, 53]

    # min_reserve_soc = [26.5, 30, 33, 36.5, 39.5]
    # target_soc_constant = [33, 40, 46, 53, 59]


    for vertiport1_, vertiport2_, network_id_, num_park_, fleet_size_, min_reserve_soc_, target_soc_constant_ in \
            product(vertiport1, vertiport2, network_id, num_park, fleet_size, min_reserve_soc, target_soc_constant):
        # print("Running simulation for: ", vertiport_layout, " with passenger arrival mu: ",
        #       passenger_arrival_mu, " aircraft arrival mu: ", aircraft_arrival_mu,
        #       " waiting time threshold: ", waiting_time_threshold, " charging time: ", charge_time)
        # Run the simulation
        loop_input(vertiport1_, vertiport2_, network_id_, num_park_, fleet_size_, min_reserve_soc_, target_soc_constant_)


if __name__ == '__main__':
    start_time = time.time()
    run_vertisim()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Total time to run experiments: {round(time_taken, 2)} seconds')


# def run_vertisim():
#     # passenger_arrival_process = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]  # [30, 60, 120, 180] # [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]  # [30, 60, 120, 180]
#     # passenger_arrival_process = [3600/16, 3600/24, 3600/36, 3600/44, 3600/52, 3600/56, 3600/64, 3600/76, 3600/84, 3600/88, 3600/96, 3600/104]
#     # passenger_arrival_process = [3600/10, 3600/20, 3600/30, 3600/40, 3600/50, 3600/60, 3600/70, 3600/80, 3600/90, 3600/100, 3600/110, 3600/120]
#     # passenger_arrival_process.extend([3600/15, 3600/25, 3600/35, 3600/45, 3600/55, 3600/65, 3600/75, 3600/85, 3600/95, 3600/105, 3600/115, 3600/125])
#     # passenger_arrival_process.sort()


#     # aircraft_arrival_process = [120, 80, 60, 48, 40, 34]
#     passenger_arrival_process = [150]
#     # aircraft_arrival_process = [60 / (6 / 60), 60 / (7 / 60), 60 / (8 / 60), 60 / (9 / 60), 60 / (10 / 60),
#     #                             60 / (11 / 60), 60 / (12 / 60), 60 / (14 / 60), 60 / (15 / 60), 60 / (16 / 60), 60 / (18 / 60), 60 / (20 / 60), 60 / (22 / 60), 60 / (24 / 60), 60 / (24 / 60)]
    
#     # aircraft_arrival_process = [60 / (4 / 60), 60 / (5 / 60), 60 / (13 / 60), 60 / (17 / 60), 60 / (19 / 60), 60 / (21 / 60), 60 / (23 / 60)]
#     # aircraft_arrival_process = [3600/1000]
#     aircraft_arrival_process = [60 / (4 / 60), 60 / (6 / 60), 60 / (8 / 60), 60 / (10 / 60),
#                             60 / (12 / 60), 60 / (14 / 60), 60 / (16 / 60),
#                             60 / (18 / 60), 60 / (20 / 60),
#                             60 / (22 / 60), 60 / (24 / 60), 60 / (26 / 60), 60 / (28 / 60), 60 / (30 / 60), 60 / (32 / 60)]
#     # aircraft_arrival_process = [60 / (4 / 60), 60 / (5 / 60), 60 / (6 / 60), 60 / (7 / 60), 60 / (8 / 60), 60 / (9 / 60), 60 / (10 / 60),
#     #                             60 / (11 / 60), 60 / (12 / 60), 60 / (13 / 60), 60 / (14 / 60), 60 / (15 / 60), 60 / (16 / 60),
#     #                             60 / (17 / 60), 60 / (18 / 60), 60 / (19 / 60), 60 / (20 / 60), 60 / (21 / 60),
#     #                             60 / (22 / 60), 60 / (23 / 60), 60 / (24 / 60)] # [45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]  # [60, 120, 180, 240, 300] [45, 75, 90, 105, 135, 150, 210]
#     max_passenger_waiting_time = [600]
#     charging_time = [60*12]
#     # charging_time = [60*16, 60*18, 60*20, 60*22, 60*24, 60*26, 60*28, 60*30] #60*1, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12, 60*14, 60*16, 60*18, 60*20, 60*22, 60*24, 60*26, 60*28, 60*30
#     # charging_time = [60*1, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12, 60*14]
#     # layouts = ['clover_1_fato_4_park']
#     layouts = ['clover_1_fato_1_park', 'clover_1_fato_2_park', 'clover_1_fato_3_park', 'clover_1_fato_4_park',
#                'clover_1_fato_5_park', 'clover_1_fato_6_park', 'clover_1_fato_7_park', 'clover_1_fato_8_park',
#                'clover_1_fato_9_park', 'clover_1_fato_10_park']
#     only_aircraft_sim = [True]
#     for passenger_arrival, aircraft_arrival_mu, waiting_time_threshold, charge_time, vertiport_layout, only_aircraft_sim_ in \
#             product(passenger_arrival_process, aircraft_arrival_process, max_passenger_waiting_time, charging_time, layouts, only_aircraft_sim):
#         # print("Running simulation for: ", vertiport_layout, " with passenger arrival mu: ",
#         #       passenger_arrival_mu, " aircraft arrival mu: ", aircraft_arrival_mu,
#         #       " waiting time threshold: ", waiting_time_threshold, " charging time: ", charge_time)
#         # Run the simulation
#         loop_input(passenger_arrival, aircraft_arrival_mu, None, None, 
#                    waiting_time_threshold, charge_time, vertiport_layout, only_aircraft_sim_)


# if __name__ == '__main__':
#     start_time = time.time()
#     run_vertisim()
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f'Total time to run experiments: {round(time_taken, 2)} seconds')


# # Only aircraft simulation
# def run_vertisim():
#     # passenger_arrival_process = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]  # [30, 60, 120, 180] # [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]  # [30, 60, 120, 180]
#     # passenger_arrival_process = [3600/10, 3600/20, 3600/30, 3600/40, 3600/50, 3600/60, 3600/70, 3600/80, 3600/90, 3600/100, 3600/110, 3600/120]
#     # aircraft_arrival_process = [120, 80, 60, 48, 40, 34]
#     passenger_arrival_process = [150]
#     aircraft_arrival_process = [60 / (4 / 60)] 
#     # 60 / (6 / 60), 60 / (7 / 60), 60 / (8 / 60), 60 / (9 / 60), 60 / (10 / 60), 60 / (11 / 60), 
#     #                             60 / (12 / 60), 60 / (14 / 60), 60 / (15 / 60), 60 / (16 / 60), 60 / (18 / 60), 60 / (20 / 60), 60 / (22 / 60), 60 / (24 / 60), 60 / (26 / 60), 60 / (30 / 60)]
#     max_passenger_waiting_time = [-1]
#     charging_time = [12*60]
#     # layouts = ['clover_1_fato_4_park']
#     layouts = ['clover_1_fato_1_park', 'clover_1_fato_2_park', 'clover_1_fato_3_park', 'clover_1_fato_4_park',
#                'clover_1_fato_5_park', 'clover_1_fato_6_park', 'clover_1_fato_7_park', 'clover_1_fato_8_park',
#                'clover_1_fato_9_park', 'clover_1_fato_10_park']
#     only_aircraft_sim = [True]
#     for passenger_arrival, aircraft_arrival_mu, waiting_time_threshold, charge_time, vertiport_layout, only_aircraft_sim_ in \
#             product(passenger_arrival_process, aircraft_arrival_process, max_passenger_waiting_time, charging_time, layouts, only_aircraft_sim):
#         # print("Running simulation for: ", vertiport_layout, " with passenger arrival mu: ",
#         #       passenger_arrival_mu, " aircraft arrival mu: ", aircraft_arrival_mu,
#         #       " waiting time threshold: ", waiting_time_threshold, " charging time: ", charge_time)
#         # Run the simulation
#         loop_input(None, None, passenger_arrival, aircraft_arrival_mu,
#                    waiting_time_threshold, charge_time, vertiport_layout, only_aircraft_sim_)


# if __name__ == '__main__':
#     start_time = time.time()
#     run_vertisim()
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f'Total time to run experiments: {round(time_taken, 2)} seconds')



# def run_vertisim():
#     combinations = pd.read_csv('../../sim_combinations.csv', index_col=[0])
#     max_passenger_waiting_time = 600
#     # charging_time = [60*1, 60*2, 60*4, 60*6, 60*8, 60*10, 60*12, 60*14, 60*16, 60*18, 60*20, 60*22, 60*24, 60*26, 60*28, 60*30]
#     layouts = 'clover_1_fato_4_park'
#     # layouts = ['clover_1_fato_1_park', 'clover_1_fato_2_park', 'clover_1_fato_3_park', 'clover_1_fato_4_park',
#     #            'clover_1_fato_5_park', 'clover_1_fato_6_park', 'clover_1_fato_7_park', 'clover_1_fato_8_park',
#     #            'clover_1_fato_9_park', 'clover_1_fato_10_park']
#     only_aircraft_sim = False
#     for passenger_arrival, aircraft_arrival_mu, charge_time in combinations.values:
#         loop_input(None, None, 3600/passenger_arrival, 3600/aircraft_arrival_mu,
#                    max_passenger_waiting_time, charge_time*60, layouts, only_aircraft_sim)


# if __name__ == '__main__':
#     start_time = time.time()
#     run_vertisim()
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f'Total time to run experiments: {round(time_taken, 2)} seconds')
