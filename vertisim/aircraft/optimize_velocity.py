import numpy as np
import sys
from scipy.optimize import minimize
from .flight_helpers import rho, weight, lift_induced_drag_coef
from .power_model import transition_power, climb_descend_power
from ..memoize import PhaseParameters

class VelocityOptimizer:
    def __init__(self, aircraft_params):
        self.aircraft_params = aircraft_params
        self.cache = {}

    @staticmethod
    def velocity(Vh, Vv):
        return np.sqrt(Vh**2 + Vv**2)

    # Define energy function
    def climb_cost(self, V, tom, altitude, vertical_dist):
        Vh, Vv = V
        climb_power = climb_descend_power(aircraft_params=self.aircraft_params, 
                                                    tom=tom,
                                                    altitude=altitude,
                                                    vertical_velocity=Vv, 
                                                    velocity=Vh, 
                                                    k_multiplier=4/3, 
                                                    direction='up')
        return climb_power * (vertical_dist/Vv)
        # Decompose the formula to improve readability
        # base_drag = 1/2 * rho(altitude=altitude, atmosphere_condition=self.aircraft_params['atmosphere_condition'])
        # velocity = VelocityOptimizer.velocity(Vh, Vv)
        # lift_coef = lift_induced_drag_coef(cd_0=self.aircraft_params['cd_0'], ld_max=self.aircraft_params['ld_max'])
        
        # power_climb = weight(tom) * Vv
        # power_drag = base_drag * velocity**3 * self.aircraft_params['wing_area'] * self.aircraft_params['cd_0']
        # power_lift = lift_coef * weight(tom)**2 / (base_drag * velocity * self.aircraft_params['wing_area'])
        
        # return (power_climb + power_drag + power_lift) * (vertical_dist/Vv) / self.aircraft_params['eta_climb']

    def descend_cost(self, V, tom, altitude, vertical_dist):
        Vh, Vv = V
        descend_power = climb_descend_power(aircraft_params=self.aircraft_params, 
                                                    tom=tom,
                                                    altitude=altitude,
                                                    vertical_velocity=Vv, 
                                                    velocity=Vh, 
                                                    k_multiplier=4/3, 
                                                    direction='down')
        return descend_power * (vertical_dist/Vv)
        # # Decompose the formula to improve readability
        # base_drag = 1/2 * rho(altitude=altitude, atmosphere_condition=self.aircraft_params['atmosphere_condition'])
        # velocity = VelocityOptimizer.velocity(Vh, Vv)
        # lift_coef = lift_induced_drag_coef(cd_0=self.aircraft_params['cd_0'], ld_max=self.aircraft_params['ld_max'])
        
        # power_descend = -weight(tom) * Vv
        # power_drag = base_drag * velocity**3 * self.aircraft_params['wing_area'] * self.aircraft_params['cd_0']
        # power_lift = lift_coef * weight(tom)**2 / (base_drag * velocity * self.aircraft_params['wing_area'])
        
        # return (power_descend + power_drag + power_lift) * (vertical_dist/Vv) / self.aircraft_params['eta_climb']
    
    def climb_transition_cost(self, V, tom, altitude, vertical_dist):
        Vh, Vv = V
        transition_start_power = transition_power(altitude=altitude, aircraft_params=self.aircraft_params, tom=tom)
        transition_end_power = climb_descend_power(aircraft_params=self.aircraft_params, 
                                                    tom=tom,
                                                    altitude=altitude,
                                                    vertical_velocity=Vv, 
                                                    velocity=Vh, 
                                                    k_multiplier=1, 
                                                    direction='up')
        return (transition_start_power + transition_end_power) / 2 * (vertical_dist/Vv)

    def descend_transition_cost(self, V, tom, altitude, vertical_dist):
        Vh, Vv = V
        transition_start_power = climb_descend_power(aircraft_params=self.aircraft_params, 
                                                    tom=tom,
                                                    altitude=altitude,
                                                    vertical_velocity=Vv, 
                                                    velocity=Vh, 
                                                    k_multiplier=1, 
                                                    direction='down')
        transition_end_power = transition_power(altitude=altitude, aircraft_params=self.aircraft_params, tom=tom)
        return (transition_start_power + transition_end_power) / 2 * (vertical_dist/Vv)   

    def min_power_velocity_for_climb_descend(self,
                                             altitude: float, 
                                             tom: float) -> float:
        """
        From the paper:
        The promise of energy-efficient battery-powered urban aircraft   

        Parameters
        ----------
        altitude : float
            altitude in m
        tom : float
            take-off mass in kg

        Returns
        -------
        float
            minimum power speed in m/s
        """
        density = rho(altitude=altitude, atmosphere_condition=self.aircraft_params['atmosphere_condition'])
        weight_term = 2 * weight(tom) / (density * self.aircraft_params['wing_area'])
        drag_term = np.sqrt(lift_induced_drag_coef(self.aircraft_params['cd_0'], self.aircraft_params['ld_max']) / (3 * self.aircraft_params['cd_0']))
        return round(np.sqrt(weight_term * drag_term))

    def max_range_speed(self, altitude, tom):
        """
        From the paper:
        The promise of energy-efficient battery-powered urban aircraft    
        """    
        density = rho(altitude=altitude, atmosphere_condition=self.aircraft_params['atmosphere_condition'])
        term1 = 2*weight(tom) / (density * self.aircraft_params['wing_area'])
        term2 = np.sqrt(lift_induced_drag_coef(cd_0=self.aircraft_params['cd_0'], ld_max=self.aircraft_params['ld_max']) / self.aircraft_params['cd_0'])    
        return round(np.sqrt(term1 * term2))     

    def set_constraints(self, V, vertical_dist, horizontal_dist):
        """ Define constraint function such that the aircraft reaches its vertical and horizontal destination at the same time.
        Ratio of Vh / Vv == D / H.
        TODO: update to use Wind class to calculate Vh_required
        """
        Vh, Vv = V
        return Vh/Vv - horizontal_dist/vertical_dist

    def optimize(self, V, tom, altitude, horizontal_distance, vertical_distance, cost_type):
        params = PhaseParameters(tom, altitude, horizontal_distance, vertical_distance, cost_type)
        if params in self.cache:
            return self.cache[params]
    
        # Map cost_type string to corresponding cost function
        cost_map = {
            'climb': self.climb_cost,
            'descend': self.descend_cost,
            'climb_transition': self.climb_transition_cost,
            'descend_transition': self.descend_transition_cost,
        }
        if vertical_distance <= 1:
            return self.max_range_speed(altitude=altitude, tom=tom), 0

        # Retrieve the appropriate cost function
        cost_function = cost_map.get(cost_type)

        # Define constraints dictionary
        constraints = {'type': 'eq', 'fun': self.set_constraints, 'args': (vertical_distance, horizontal_distance)}

        # Define bounds for V and Vv
        min_power_velocity = self.min_power_velocity_for_climb_descend(altitude=altitude, tom=tom)
        if cost_type in ['climb_transition', 'descend_transition']:
            bounds = [(round(min_power_velocity/2, 2), min_power_velocity), 
                    (1, self.aircraft_params['max_vertical_velocity'])]
        else:
            bounds = [(round(min_power_velocity/2), self.aircraft_params['max_horizontal_velocity']), 
                    (1, self.aircraft_params['max_vertical_velocity'])]

        # Initial guess for V and Vv
        x0 = [V[0], V[1]]

        # Perform optimization
        result = minimize(cost_function, x0, args=(tom, altitude, vertical_distance), method='SLSQP', bounds=bounds, constraints=constraints)

        vHO, vVO = round(result.x[0], 2), round(result.x[1], 2)
        self.cache[params] = (vHO, vVO)

        return vHO, vVO            

