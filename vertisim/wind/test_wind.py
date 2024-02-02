import unittest
import numpy as np
from ..wind.wind import Wind
from ..utils.vectorMath import vector_to_heading, swap_angle_relative_x_axis_north, magnitude, convert_to_range_zero_to_two_pi
from math import pi, cos, sin

class TestWind(unittest.TestCase):
    
    def test_make_wind_vector_relative_to_north(self):
        v_wind = Wind.make_wind_vector_relative_to_north(pi / 6, wind_magnitude=10.0) # S + 30 degrees W wind, 10.0 m / s 
        self.assertAlmostEqual(v_wind[0], -10 * sin(pi / 6), msg="Incorrect wind vector x-component.")
        self.assertAlmostEqual(v_wind[1], -10 * cos(pi / 6), msg="Incorrect wind vector y-component.")
        
    def test_magnitude(self):
        v = np.array([3, 4])
        mag = magnitude(v)
        self.assertAlmostEqual(mag, 5)
        
    def test_ground_v(self):
        v_wind = Wind.make_wind_vector_relative_to_north(pi / 4, wind_magnitude=10.0) # SW wind, 10.0 m / s 
        true_air_v = np.array([0, 50.0]) # north, 50.0 m / s
        gv = Wind.ground_velocity(true_air_v, v_wind)
        self.assertAlmostEqual(gv[0], -10 * sin(pi / 4), msg="Incorrect ground velocity x-component.")
        self.assertAlmostEqual(gv[1], 50 - 10 * cos(pi / 4), msg="Incorrect ground velocity y-component.")

    def test_vector_to_heading(self):
        # angle 1
        actual_heading = pi / 3 # from North
        vector = np.array([sin(actual_heading),
                           cos(actual_heading)]) # unit vector
        returned_heading = vector_to_heading(vector)
        self.assertAlmostEqual(returned_heading, actual_heading, msg="Heading incorrect.")
        # angle 2
        actual_heading = 3 * pi / 4 # from North
        vector = np.array([sin(actual_heading),
                           cos(actual_heading)]) # unit vector
        returned_heading = vector_to_heading(vector)
        self.assertAlmostEqual(returned_heading, actual_heading, msg="Heading incorrect.")
        # angle 3
        actual_heading = 7 * pi / 4 # from North
        vector = np.array([sin(actual_heading),
                           cos(actual_heading)]) # unit vector
        returned_heading = vector_to_heading(vector)
        self.assertAlmostEqual(returned_heading, actual_heading, msg="Heading incorrect.")

    def test_swap_angle_relative_x_axis_north(self):
        # angle_relative_x < pi / 2
        angle_relative_x = 3 * pi / 8
        angle_relative_N = (pi / 2) - angle_relative_x
        returned_angle_relative_N = swap_angle_relative_x_axis_north(angle_relative_x)
        self.assertAlmostEqual(angle_relative_N, returned_angle_relative_N, msg="Angle conversion failed.")
        # angle_relative_x  > pi / 2
        angle_relative_x = 7 * pi / 4
        angle_relative_N = 3 * pi / 4
        returned_angle_relative_N = swap_angle_relative_x_axis_north(angle_relative_x)
        self.assertAlmostEqual(angle_relative_N, returned_angle_relative_N, msg="Angle conversion failed.")
        # N --> x conversion
        angle_relative_x = 7 * pi / 4
        angle_relative_N = 3 * pi / 4
        returned_angle_relative_x = swap_angle_relative_x_axis_north(angle_relative_N)
        self.assertAlmostEqual(angle_relative_x, returned_angle_relative_x, msg="Angle conversion failed.")

    def test_wind_adjusted_true_velocity(self):
        """Test all possible desired headings, wind headings, and a range of cruise and wind speeds.
        Completes in about 50 seconds."""
        
        for cruise_speed in np.arange(10, 60, 1):
            for wind_mag in np.arange(0, 20, 1):
                for desired_heading in np.arange(0, 2*pi, 10*pi/180):
                    for wind_heading in np.arange(0, 2*pi, 10*pi/180):
                        
                        # can use this to set a conditional break point based on output
                        parameters = f'Parameters: cruise_speed == {cruise_speed} and wind_mag == {wind_mag} and desired_heading == {desired_heading} and wind_heading == {wind_heading}'
                        
                        ground_speed_threshold = 30.0 # m / s

                        v_wind = Wind.make_wind_vector_relative_to_north(wind_heading, wind_magnitude=wind_mag)
                        
                        true_velocity = Wind.wind_adjusted_true_velocity(cruise_speed, desired_heading, v_wind, ground_speed_threshold=ground_speed_threshold)
                        true_airspeed = magnitude(true_velocity)
                        
                        ground_velocity = Wind.ground_velocity(true_velocity, v_wind)
                        ground_speed = magnitude(ground_velocity)
                        ground_heading = convert_to_range_zero_to_two_pi(vector_to_heading(ground_velocity))
                        
                        ground_data = f'Ground data: ground_heading = {ground_heading}, ground_speed = {ground_speed}'
                        
                        data = '\n' + parameters + '\n' + ground_data

                        self.assertAlmostEqual(ground_heading, desired_heading, msg=f'\nAircraft is not flying in the right direction. ' + data)
                        if ground_speed < ground_speed_threshold:
                            self.assertAlmostEqual(ground_speed, ground_speed_threshold, msg=f'\nAircraft\'s ground speed is below the threshold. ' + data)
                        if not abs(ground_speed - ground_speed_threshold) < 1e-9: 
                            self.assertAlmostEqual(true_airspeed, cruise_speed, msg=f'\nAircraft does not have true_airspeed = cruise_speed.' + data)
        
if __name__ == '__main__':
    unittest.main()