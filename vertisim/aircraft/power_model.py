from .flight_helpers import rho, weight, temperature, lift_induced_drag_coef, rotor_disk_area, G_CONSTANT
import numpy as np

def transition_power(altitude, aircraft_params, tom):
    """
    Returns transition start or end power in kW
    """
    density = rho(altitude=altitude, atmosphere_condition=aircraft_params['atmosphere_condition'])

    term1 = aircraft_params['f'] * weight(tom) / aircraft_params['FoM']
    term2 = np.sqrt(aircraft_params['f'] * weight(tom) / rotor_disk_area(tom, aircraft_params['disk_load']) / (2 * density))

    return (term1 * term2) / aircraft_params['eta_hover']

def climb_descend_power(aircraft_params, tom, altitude, vertical_velocity, velocity, k_multiplier=1, direction='up'):
    """
    Returns general climb power in kW
    """
    if direction == 'down':
        vertical_velocity = -vertical_velocity
    elif direction != 'up':
        raise ValueError('direction must be either "up" or "down"')
    density = rho(altitude=altitude, atmosphere_condition=aircraft_params['atmosphere_condition'])
    term1 = weight(tom) * vertical_velocity
    term2 = 1/2 * density * aircraft_params['wing_area'] * aircraft_params['cd_0'] * velocity**3
    term3 = k_multiplier*lift_induced_drag_coef(cd_0=aircraft_params['cd_0'], ld_max=aircraft_params['ld_max']) \
            * weight(tom)**2 / (1/2*density * aircraft_params['wing_area'] * velocity)
    return (term1 + term2 + term3) / aircraft_params['eta_hover']

def vertical_takeoff_landing_phase_power(start_altitude, end_altitude, aircraft_params, tom, vertical_velocity):
    start_density = rho(altitude=start_altitude, atmosphere_condition=aircraft_params['atmosphere_condition']) # This doesn't matter whether the operation is takeoff or landing as we take the average of the two.
    end_density = rho(altitude=end_altitude, atmosphere_condition=aircraft_params['atmosphere_condition'])

    term1 = aircraft_params['f'] * weight(tom) / aircraft_params['FoM']
    term2_start = np.sqrt(aircraft_params['f'] * weight(tom) / rotor_disk_area(tom, aircraft_params['disk_load']) / (2 * start_density))
    term2_end = np.sqrt(aircraft_params['f'] * weight(tom) / rotor_disk_area(tom, aircraft_params['disk_load']) / (2 * end_density))
    term3 = weight(tom) * vertical_velocity / 2

    start_power = max((term1 * term2_start + term3) / aircraft_params['eta_hover'], 0)
    end_power = max((term1 * term2_end + term3) / aircraft_params['eta_hover'], 0)

    return (start_power + end_power)/2

def climb_transition_phase_power(start_altitude, end_altitude, aircraft_params, tom, vertical_velocity, velocity):
    climb_transition_start_power = transition_power(altitude=start_altitude, aircraft_params=aircraft_params, tom=tom)
    climb_transition_start_power = max(climb_transition_start_power, 0)
    climb_transition_end_power = climb_descend_power(aircraft_params, tom, end_altitude, vertical_velocity, velocity)
    climb_transition_end_power = max(climb_transition_end_power, 0)
    return (climb_transition_start_power + climb_transition_end_power)/2

def climb_phase_power(start_altitude, end_altitude, aircraft_params, tom, vertical_velocity, velocity):
    k_multiplier = 4/3  # New K to account for L/D correction
    climb_start_power = climb_descend_power(aircraft_params, tom, start_altitude, vertical_velocity, velocity, k_multiplier=k_multiplier)
    climb_start_power = max(climb_start_power, 0)
    climb_end_power = climb_descend_power(aircraft_params, tom, end_altitude, vertical_velocity, velocity, k_multiplier=k_multiplier)
    climb_end_power = max(climb_end_power, 0)
    return (climb_start_power + climb_end_power)/2
    
def cruise_phase_power(cruise_speed, aircraft_params, tom):
    return max((weight(tom) * cruise_speed) / (0.85*aircraft_params['ld_max']) / aircraft_params['eta_cruise'], 0)

def descend_phase_power(start_altitude, end_altitude, aircraft_params, tom, vertical_velocity, velocity):
    k_multiplier = 4/3  # New K to account for L/D correction
    descend_start_power = climb_descend_power(aircraft_params, tom, start_altitude, vertical_velocity, velocity, k_multiplier=k_multiplier, direction='down')
    descend_start_power = max(descend_start_power, 0)
    descend_end_power = climb_descend_power(aircraft_params, tom, end_altitude, vertical_velocity, velocity, k_multiplier=k_multiplier, direction='down')
    descend_end_power = max(descend_end_power, 0)
    return (descend_start_power + descend_end_power)/2

def descend_transition_phase_power(start_altitude, end_altitude, aircraft_params, tom, vertical_velocity, velocity):
    descend_transition_start_power = climb_descend_power(aircraft_params=aircraft_params, 
                                                         tom=tom, altitude=start_altitude, 
                                                         vertical_velocity=vertical_velocity, 
                                                         velocity=velocity, 
                                                         direction='down')
    descend_transition_start_power = max(descend_transition_start_power, 0)
    descend_transition_end_power = transition_power(altitude=end_altitude, aircraft_params=aircraft_params, tom=tom)
    descend_transition_end_power = max(descend_transition_end_power, 0)
    return (descend_transition_start_power + descend_transition_end_power)/2