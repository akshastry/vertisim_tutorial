import time
from sim_setup import SimSetup
import os
from pathlib import Path
import re
from math import pi

cwd = Path.cwd()

# ------------------------ Inputs ------------------------
# Simulation parameters
# --------------------
sim_params = {
    'sim_time': 60 * 60 * 60,  # seconds
    'arrival_priority': 1,  # Lower the value higher the priority
    'departure_priority': 1,  # Lower the value higher the priority
    'simultaneous_taxi_and_take_off': True,
    'num_initial_passengers': 0,   # TODO: # of passengers at the vertiport at the beginning of the simulation.
    'random_seed': 42,
    'max_passenger_waiting_time': 600,  # seconds or None
    'flight_duration_constant': None,  # seconds or None - Applies ground holding if the flight duration is less than this value.
    'charge_interruption': False, #TODO: Not supported yet.
    'charge_assignment_sensitivity': 2.4, # WILL REMOVE THIS PARAMETER.
    'step_by_step_sim': True,
    'step_by_step_sim_response_type': 'auto', # ['hil', 'auto']. HIL stops the simulation at each event and waits input from user to continue 
                                              # 'auto' stops the simulation at the specific events and dumps the states. 
                                              # 'auto' can be used by the external optimization tools.
    'simulation_states': ['num_aircraft', 'num_flight_requests', 'num_available_fato', 'num_available_parking_pad', 
                          'num_waiting_passengers', 'total_passenger_waiting_time', 'arrival_fix', 
                          'origin_vertiport_id', 'destination_vertiport_id', 'soc',
                          'time_to_arrival', 'status', 'wind_speed', 'wind_direction'], # For every vertiport, aircraft, and airspace node
                          # These are the states that will be returned by the simulator if any external applications
                          # (AI and Optimization algorithms) that use the simulator as the environment.
                          # If not needed, keep it None.
    'fleet_rebalancing': True,
    'save_trajectories': True, # If True, then the simulator will save the trajectories of the aircraft and passengers.
    'only_aircraft_simulation': False,  # If True, then the simulation will only simulate aircraft turnaround.
    'network_simulation': True,  # If True, then the simulation will simulate the network.
    'verbose': True  # If True, then the simulation will print the simulation output and performance metrics.
    # 'aircraft_assignment_rule_1': True,  # If True, then the simulator will use rule 1 for aircraft assignment. (
    #                                      # Assign the least capacity aircraft that can complete the mission)
    # 'aircraft_assignment_rule_2': False  # If True, then the simulator will use rule 2 for aircraft assignment. (
    #                                      # Fill the aircraft that can complete the mission and arrived first)
}

# Network and Trips
# -----------------
network_and_demand_params = {
    'vertiport_network_file_path': f'{str(cwd)}/vertisim/input/network/network_SFO_MV.csv',
    'vertiport_layout_file_path': f'{str(cwd)}/vertisim/input/vertiport_layouts/vertiport_layout.xls',

    # DEMAND FILES.
    # ---------------------------------------------------------------
    # If these are not None, then the demand is read from these files.
    # If these are None, then the artificial demand is generated. Priority is given to the demand files.
    # If charge_schedule_file_path is None, then the flight_schedule should provide the tail_number and departure times.
    # Charging times will be computed internally. If charge_schedule_file_path is not None, then the charging times will be read from the file.
    # and aircraft assignment will be done based on the optimization output.
    'charge_schedule_file_path': None, #f'{str(cwd)}/input/aircraft/charge_schedule.csv',
    'flight_schedule_file_path': None, #f'{str(cwd)}/input/aircraft/flight_schedule.csv',
    'passenger_schedule_file_path': None, #f'{str(cwd)}/input/passenger/demand_SFO_MV.csv',  # Use a full path. If None, passenger_arrival_generator will be used. 
    'passenger_arrival_rates_path': f'{str(cwd)}/vertisim/input/passenger/passenger_arrival_rates_sfo_mv.csv',  # Use a full path. 

    # If the demand files are None, then the artificial demand is generated.
    'demand_probabilities': None, #[0.5, 0.5], # The probability of the demand to be generated for each vertiport.

    'vertiports': {
    # Input specifications for each vertiport.
        'SFO': {
            'layout': 'clover_1_fato_8_park_sfo',
            'holding_unit_capacity': 20,            
            'num_security_check_servers' : 10,
            'num_chargers': 8,
            'charger_max_charge_rate': 350,  # kW
            'charger_efficiency': 0.90,
            'shared_charger_sets': None,
            # # Note: PARK IDs should match with the excel/csv file PARK IDs.
            # 'shared_charger_sets' : {
            #     1: ['LAX_PARK11', 'LAX_PARK12'],
            #     2: ['LAX_PARK12', 'LAX_PARK13'],
            #     3: ['LAX_PARK13', 'LAX_PARK14']
            # },

            # ARTIFICIAL PASSENGER DEMAND GENERATION
            # --------------------------------------
            # Prob. dist. name and the parameter variable names should be consistent with the reference:
            # https://docs.scipy.org/doc/scipy/reference/stats.html. You may need to add/remove some parameters
            # depending on the distribution you use.
            'passenger_arrival_process': {
                'passenger_interarrival_constant': None,  # If not None, then the passenger arrival process will be created by the constant.
                'num_passengers': 1000,  # Number of passengers to be generated
                'passenger_arrival_distribution': {
                    'distribution_name': 'expon',
                    'parameters': {
                        'scale': None  # seconds
                        # 'param3': None,
                        # 'param4': None
                    },
                    'max_val_in_dist': None,  # seconds
                },
            },
            # ARTIFICIAL AIRCRAFT SUPPLY GENERATION
            # -------------------------------------
            'aircraft_arrival_process': {
                'num_initial_aircraft_at_vertiport':5,  # Number of aircraft in the simulation/vertiport. 
                'initial_arrival_queue_state': 0,
                # For network simulation num_aircraft cannot exceed the number of parking pads on that vertiport. 
                # In the case of single vertiport sim, if num_aircraft None, then the required number of aircraft will be estimated from the num_passengers.
                # For network simulatio, if num_aircraft is None, then the  num_aircraft will be half of the parking pads.
                
                # THERE REST OF THE PARAMETERS ARE FOR SINGLE VERTIPORT SIMULATION.
                # 'aircraft_interarrival_constant': None,  # If not None, then the aircraft arrival process will be created by the constant.
                # 'num_aircraft': 200,  # Number of aircraft to be generated
                # 'num_passengers': None,  # If num_aircraft is defined, this can be None because num_aircraft will overwrite. 
                # 'aircraft_arrival_distribution': {
                #     'distribution_name': 'expon',
                #     'parameters': {
                #         'scale': 3600/14  # seconds
                #         # 'param3': None,
                #         # 'param4': None
                #     },
                #     'max_val_in_dist': None,  # seconds
                # }
            }
        },
        'MV': {
            'layout': 'clover_1_fato_8_park_mv',
            'holding_unit_capacity': 20,
            'num_security_check_servers' : 10,
            'num_chargers': 8,
            'charger_max_charge_rate': 350,  # kW
            'charger_efficiency': 0.90,
            'shared_charger_sets': None,
            # Passenger Arrival Process
            'passenger_arrival_process': {
                'passenger_interarrival_constant': None,
                'num_passengers': 1000,  
                'passenger_arrival_distribution': {
                    'distribution_name': 'expon',
                    'parameters': {
                        'scale': None  # seconds
                    },
                    'max_val_in_dist': None,  # seconds
                },
            },
            # Aircraft Arrival Process
            'aircraft_arrival_process': {
                'num_initial_aircraft_at_vertiport': 8,
                'initial_arrival_queue_state': 0
            }
        }     
    }
}

# Airspace parameters
# -------------------
airspace_params = {
    'airspace_layout_file_path': None, #f'{str(cwd)}/vertisim/input/airspace/finalized_route_0615.csv',
    'airspace_layout_sheet_name': None,
    'airlink_capacity': 1,
    # If the airlink file is not None, then the below parameters are not used.
    'airlink_segment_length_mile': 1,  # miles
    'holding_unit_capacity': 20, # TODO: Remove this from here. This is vertiport parameter.
    'cruise_altitude': 450,  # m
    'static_wind': False, # If True, then wind data won't be used. Instead the wind data will be generated based on the wind parameters below.
    'wind_magnitude': 8,  # m/s
    'wind_angle': pi,  # deg - Relative to the aircraft heading
    'wind_data_file_path': f'{str(cwd)}/vertisim/input/airspace/SFBayArea_wind_data.csv'
}

# Passenger parameters
# --------------------
# Time parameters should be a constant or a distribution. One should be None, the other should be provided.
passenger_params = {
    'randomize_constants': False,  # If true, the constants will be randomized. If false, the constants will be used as
    # it is.
    'car_to_entrance_walking_time_constant': 0, #30,  # secs
    'car_to_entrance_walking_time_dist': None,
    'security_check_time_constant': 0, #30,  # secs
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
    'waiting_room_to_boarding_gate_walking_time_constant': 0, #30,  # secs
    'waiting_room_to_boarding_gate_walking_time_dist': None,
    'boarding_gate_to_aircraft_time_constant': 0, #19.7,  # secs
    'boarding_gate_to_aircraft_time_dist': None,
    'deboard_aircraft_and_walk_to_exit_constant': 0, #120,  # secs
    'deboard_aircraft_and_walk_to_exit_dist': None,
}

# Aircraft parameters
# -------------------
aircraft_params = {
    'aircraft_model': 'jobyS4',  # This is the default aircraft type, if aircraft type is given with the schedule,
    # then that will be used as the aircraft type.
    'pax': 4,  # Default passenger capacity, if passenger capacity is given with the schedule,
    # then that will be used as the passenger capacity.
    'range': 150,  # miles
    'soc': 100,  # Default initial SOC of aircraft at the arrival to the vertiport
    'battery_capacity': 160,  # kWh
    'pax_mass': 100,  # kg
    
    # MISSION PARAMETERS
    # -----------------
    'vertical_takeoff_velocity': 1.5,  # m/s
    'vertical_landing_velocity': 1.5,  # m/s
    'climb_transition_end_forward_velocity': 18,  # m/s
    'climb_transition_vertical_velocity': 1.5,  # m/s
    # 'initial_climb_phase_end_forward_velocity': 45,  # m/s
    # 'initial_climb_phase_vertical_velocity': 3.125,  # m/s
    'climb_phase_end_forward_velocity': 45,  # m/s
    'climb_phase_vertical_velocity': 3.125,  # m/s
    'cruise_speed': 67, #30, # or None
    # 'initial_descend_phase_end_forward_velocity': 36,  # m/s
    # 'initial_descend_phase_vertical_velocity': 3,  # m/s
    'descend_phase_end_forward_velocity': 36,  # m/s
    'descend_phase_vertical_velocity': 3.8,  # m/s
    'descend_transition_end_forward_velocity': 0,  # m/s
    'descend_transition_vertical_velocity': 3.8,  # m/s

    'max_vertical_velocity': 5,  # m/s
    'max_horizontal_velocity': 89,  # m/s - 173 knots | 200 mph
    # # Altitude
    'ground_altitude': 0,  # m
    'hover_altitude': 15,  # m
    'cruise_altitude': 450,  # m

    # Process time
    'time_tug_connection': 0,  # sec
    'time_tug_disconnection': 0,  # sec
    'time_pre_take_off_check_list': 0,  # sec
    'time_rotor_spin_up': 5,  # sec - included in the hover time. Doesn't consume energy
    'time_hover_climb': 15,  # sec - NOT USED
    'time_climb_transition': 30,  # sec - NOT USED
    'time_climb': 65,  # sec - NOT USED  
    'time_descend': 65,  # sec - NOT USED
    'time_descend_transition': 30,  # sec - NOT USED
    'time_hover_descend': 15,  # sec - NOT USED
    'time_rotor_spin_down': 5,  # sec - included in the hover time. Doesn't consume energy
    'time_post_landing_safety_checks': 0,  # sec    
    'ground_taxi_speed': 3.67, # 3.67 # ft/s
    'time_passenger_embark_disembark': 60 * 2, #60 * 2,  # secs
    'time_pre_charging_processes': 60 * 3,  # sec
    'time_charging_plug_disconnection': 0,  # sec
    'time_post_charging_processes': 60 * 3,  # sec'


    # AERODYNAMICS PARAMETERS
    # -----------------------
    'mtom': 2182,  # kg
    'wing_area': 13,  # m^2
    'disk_load': 45.9,  # kg/m^2
    'f': 1.03, # Correction factor for interference from the fuselage
    'FoM': 0.8,  # Figure of merit
    'cd_0': 0.015,  # Zero lift drag coefficient
    'cl_max': 1.5,  # Maximum lift coefficient
    'ld_max': 18,  # Maximum lift to drag ratio
    'eta_hover': 0.85,  # Hover efficiency
    'eta_climb': 0.85,  # Climb efficiency
    'eta_descend': 0.85,  # Descend efficiency
    'eta_cruise': 0.90,  # Cruise efficiency
    'atmosphere_condition': 'good', # good, bad


    # AIRCRAFT CHARGING TIME PARAMETERS
    # ----------------------------
    # There are 3 options to simulate the charging time of an aircraft:
    # 1. Constant charging time defined by 'time_charging' parameter.
    # 2. Charging time distribution defined by 'time_charging_dist' parameter.
    # 3. Based on energy requirement to complete the mission. Each aircraft type has a different energy requirement to
    # complete the mission. To use this feature, battery model parameters and aircraft energy consumption data need to
    # be given. The energy requirement is computed with the model that Sridhar et. al. developed. TODO: Add link

    'time_charging': None,  # constant sec. If None charging_time_dist will be used.
    'target_soc_constant': 60,  # Target SOC of the aircraft for each charging event
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
    'aircraft_energy_consumption_data_folder_path': f'{str(cwd)}/vertisim/input/aircraft/energy_consumption',
    'min_init_soc': 20,  # Minimum initial SOC of the aircraft at the arrival to the vertiport
    'max_init_soc': 100,  # Maximum initial SOC of the aircraft at the arrival to the vertiport
    'min_reserve_soc': 40  # This is the minimum acceptable SoC that the aircraft should have to complete the mission.
}

output_params = {
    'output_folder_path': f'{str(cwd.parent)}/output/results',
    'config_output_file_name': 'config',
    'performance_metrics_output_file_name': 'performance_metrics'
}
