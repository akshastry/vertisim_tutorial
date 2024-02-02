import os
cwd = os.getcwd()

DEFAULT_SIM_PARAMS = {
    'sim_time': 60 * 60 * 100,  # seconds
    'arrival_priority': 1,  # Lower the value higher the priority
    'departure_priority': 1,  # Lower the value higher the priority
    'simultaneous_taxi_and_take_off': True,
    'num_initial_passengers': 0,   # TODO: # of passengers at the vertiport at the beginning of the simulation.
    'random_seed': 42,
    'max_passenger_waiting_time': 600,  # seconds
    'charge_interruption': False,
    'save_trajectories': True, # If True, then the simulator will save the trajectories of the aircraft and passengers.
    'only_aircraft_simulation': False,  # If True, then the simulation will only simulate aircraft turnaround.
    'network_simulation': True,  # If True, then the simulation will simulate the network.
    'verbose': True,  # If True, then the simulation will print the simulation output and performance metrics.
    'battery_model_sim': False  # If True, then the simulation will use the battery model.
    # 'aircraft_assignment_rule_1': True,  # If True, then the simulator will use rule 1 for aircraft assignment. (
    #                                      # Assign the least capacity aircraft that can complete the mission)
    # 'aircraft_assignment_rule_2': False  # If True, then the simulator will use rule 2 for aircraft assignment. (
    #                                      # Fill the aircraft that can complete the mission and arrived first)
}

# Network and Trips
# -----------------
DEFAULT_NETWORK_AND_DEMAND = {
    'vertiport_network_file_path': f'{cwd}/vertisim/input/network_of_2_vertiports.csv',
    'vertiport_layout_file_path': f'{cwd}/vertisim/input/skyport_layouts/skyport_layout.xls',

    # DEMAND FILES.
    # ---------------------------------------------------------------
    # If these are not None, then the demand is read from these files.
    # If these are None, then the artificial demand is generated.
    # Priority is given to the demand files.
    'flight_schedule_file_path': None,  # Use a full path. Fixed schedule simulation. If None, aircraft_arrival_generator will be used.
    'passenger_schedule_file_path': None,  # Use a full path. If None, passenger_arrival_generator will be used. 

    # If the demand files are None, then the artificial demand is generated.
    'demand_probabilities': [0.5, 0.5], # The probability of the demand to be generated for each vertiport.
}

DEFAULT_VERTIPORT_PARAMS = {
            'layout': 'clover_1_fato_4_park_lax',
            'holding_unit_capacity': 20,            
            'num_security_check_servers' : 4,
            'num_chargers': 4,
            'charger_max_charge_rate': 480,  # kW
            'charger_efficiency': 0.90,
            # 'shared_charger_sets': None,
            # # Note: PARK IDs should match with the excel/csv file PARK IDs.
            'shared_charger_sets' : {
                1: ['LAX_PARK11', 'LAX_PARK12'],
                2: ['LAX_PARK12', 'LAX_PARK13'],
                3: ['LAX_PARK13', 'LAX_PARK14']
            }
}

            # ARTIFICIAL PASSENGER DEMAND GENERATION
            # --------------------------------------
DEFAULT_PASSENGER_ARRIVAL_PROCESS = {
                'passenger_interarrival_constant': None,  # If not None, then the passenger arrival process will be created by the constant.
                'num_passengers': 200,  # Number of passengers to be generated
                'passenger_arrival_distribution': {
                    'distribution_name': 'expon',
                    'parameters': {
                        'scale': 3600/60  # seconds
                        # 'param3': None,
                        # 'param4': None
                    },
                    'max_val_in_dist': None,  # seconds
                }
}
            # ARTIFICIAL AIRCRAFT SUPPLY GENERATION
            # -------------------------------------
DEFAULT_AIRCRAFT_ARRIVAL_PROCESS = {
                'num_initial_aircraft_at_vertiport': 4,  # Number of aircraft in the simulation/vertiport. 
                'initial_parking_occupancy': 4,  # The # of the parking spaces that are occupied with aircraft at the beginning of the sim.
                'initial_arrival_queue_state': 0,  # TODO: Number of aircraft in the arrival queue at the beginning of the sim.                
                'num_aircraft': 200,  # Number of aircraft to be generated
                'num_passengers': None,  # If num_aircraft is defined, this can be None because num_aircraft will overwrite. 
                'aircraft_arrival_distribution': {
                    'distribution_name': 'expon',
                    'parameters': {
                        'scale': 3600/14  # seconds
                    },
                    'max_val_in_dist': None,  # seconds
                }
}

# Airspace parameters
# -------------------
DEFAULT_AIRSPACE_PARAMS = {
    'airspace_layout_file_path': None,
    'airlink_capacity': 1,
    'airlink_segment_length_mile': 0.25  # miles
}



# Passenger parameters
# --------------------
# Time parameters should be a constant or a distribution. One should be None, the other should be provided.
DEFAULT_PASSENGER_PARAMS = {
    'randomize_constants': False,  # If true, the constants will be randomized. If false, the constants will be used as
    # it is.
    'car_to_entrance_walking_time_constant': 30,  # secs
    'car_to_entrance_walking_time_dist': None,
    'security_check_time_constant': 30,  # secs
    'security_check_time_dist': None,
    # 'security_check_time_dist': {
    #     'distribution_name': 'expon',
    #     'parameters': {
    #         'loc': 0,
    #         'scale': 1
    #         # 'param3': None,
    #         # 'param4': None
    #     },
    #     'max_val_in_dist': 60
    # },
    'waiting_room_to_boarding_gate_walking_time_constant': 31.9,  # secs
    'waiting_room_to_boarding_gate_walking_time_dist': None,
    'boarding_gate_to_aircraft_time_constant': 19.7,  # secs
    'boarding_gate_to_aircraft_time_dist': None,
    'deboard_aircraft_and_walk_to_exit_constant': 120,  # secs
    'deboard_aircraft_and_walk_to_exit_dist': None,
}

# Aircraft parameters
# -------------------
DEFAULT_AIRCRAFT_PARAMS = {
    'aircraft_model': 'jobyS4',  # This is the default aircraft type, if aircraft type is given with the schedule,
    # then that will be used as the aircraft type.
    'pax': 4,  # Default passenger capacity, if passenger capacity is given with the schedule,
    # then that will be used as the passenger capacity.
    'range': 150,  # miles
    'soc': 20,  # Default initial SOC of aircraft at the arrival to the vertiport
    'battery_capacity': 160,  # kWh
    
    # TAKE-OFF, LANDING, AND TURNAROUND TIME PARAMETERS
    # ------------------------------------------------
    'ground_taxi_speed': 3.67,  # ft/s
    'time_passenger_embark_disembark': 60 * 0,  # secs
    'time_pre_charging_processes': 0,  # sec
    'time_charging_plug_disconnection': 0,  # sec
    'time_post_charging_processes': 0,  # sec
    'time_descend_transition': 60,  # sec
    'time_hover_descend': 0,  # sec
    'time_rotor_spin_down': 0,  # sec
    'time_post_landing_safety_checks': 0,  # sec
    'time_tug_connection': 0,  # sec
    'time_tug_disconnection': 0,  # sec
    'time_pre_take_off_check_list': 0,  # sec
    'time_rotor_spin_up': 0,  # sec
    'time_hover_climb': 0,  # sec
    'time_climb_transition': 60,  # sec

    # FLIGHT SPEED PARAMETERS
    # -----------------
    'cruise_speed': 120,  # mph
    'vertical_takeoff_velocity': 5.5,  # mph # TODO
    'vertical_landing_velocity': 5.5,  # mph
    'climb_speed': 60,  # mph
    'descent_speed': 60,  # mph



    # AIRCRAFT CHARGING TIME PARAMETERS
    # ----------------------------
    # There are 3 options to simulate the charging time of an aircraft:
    # 1. Constant charging time defined by 'time_charging' parameter.
    # 2. Charging time distribution defined by 'time_charging_dist' parameter.
    # 3. Based on energy requirement to complete the mission. Each aircraft type has a different energy requirement to
    # complete the mission. To use this feature, battery model parameters and aircraft energy consumption data need to
    # be given. The energy requirement is computed with the model that Sridhar et. al. developed. TODO: Add link

    'time_charging': None,  # constant sec. If None charging_time_dist will be used.
    # Prob. dist. name and the parameter variable names should be consistent with the reference:
    # https://docs.scipy.org/doc/scipy/reference/stats.html. You may need to add/remove some parameters
    # depending on the distribution you use.
    'charging_time_dist': {
        'distribution_name': 'expon',
        'parameters': {
            'scale': 60 * 10
            # 'param3': None,
            # 'param4': None
        },
        'max_val_in_dist': None,  # sec. This parameter is used as the max charge time for the charging
        # time distribution generation.
    },
    'aircraft_energy_consumption_data_folder_path': f'{cwd}/input/aircraft_energy_consumption',
    'min_init_soc': 20,  # Minimum initial SOC of the aircraft at the arrival to the vertiport
    'max_init_soc': 100,  # Maximum initial SOC of the aircraft at the arrival to the vertiport
    'min_reserve_soc': 20  # This is the minimum acceptable SoC that the aircraft should have at the end of the mission.
}

DEFAULT_OUTPUT_PARAMS = {
    'output_folder_path': f'{cwd}/output/test_only_aircraft',
    'config_output_file_name': 'config',
    'performance_metrics_output_file_name': 'performance_metrics',
}