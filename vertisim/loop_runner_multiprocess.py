from loop_input import *
from multiprocessing import Process


def run_loops(passenger_arrival_process_,
              aircraft_arrival_process_,
              max_passenger_waiting_time_,
              charging_time_,
              layouts_):
    processes = []
    # Create a process for each vertisim simulation
    for passenger_arrival_mu in passenger_arrival_process_:
        for aircraft_arrival_mu in aircraft_arrival_process_:
            for waiting_time_threshold in max_passenger_waiting_time_:
                for charge_time in charging_time_:
                    for vertiport_layout in layouts_:
                        p = Process(target=loop_input, args=(passenger_arrival_mu, aircraft_arrival_mu,
                                                             waiting_time_threshold, charge_time, vertiport_layout))
                        processes.append(p)
                        p.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == '__main__':

    passenger_arrival_process = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    aircraft_arrival_process = [60, 90, 120, 150, 180, 210, 240, 270, 300]
    max_passenger_waiting_time = [600, 60 * 30]
    charging_time = [60 * 5, 60 * 7, 60 * 10, 60 * 15]
    layouts = ['clover_1_fato_1_park', 'clover_1_fato_2_park', 'clover_1_fato_3_park', 'clover_1_fato_4_park',
               'clover_1_fato_5_park', 'clover_1_fato_6_park', 'clover_1_fato_7_park', 'clover_1_fato_8_park']

    start_time = time.time()
    run_loops(passenger_arrival_process,
              aircraft_arrival_process,
              max_passenger_waiting_time,
              charging_time,
              layouts)

    end_time = time.time()
    time_taken = end_time - start_time
    print(
        f'Total time to run {len(passenger_arrival_process) * len(aircraft_arrival_process) * len(max_passenger_waiting_time) * len(charging_time) * len(layouts)} experiments: {round(time_taken, 2)} seconds')
