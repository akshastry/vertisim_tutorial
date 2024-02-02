import numpy as np
from math import atan2, pi, sin, cos

def vector_to_heading(vector: np.ndarray) -> float:
    """Returns the angle of the vector relative to the positive Y-axis (North) in radians."""
    
    # angle relative to x axis
    angle_from_x = atan2(vector[1], vector[0])
    
    return swap_angle_relative_x_axis_north(angle_from_x) 

def swap_angle_relative_x_axis_north(angle: float) -> float:
    """Converts an angle (radians) relative to the x-axis (CCW = pos) to be relative to North (CW = pos), and vice versa."""
    if angle <= pi / 2:
        return (pi / 2) - angle
        
    return (5 * pi / 2) - angle

def lat_long_to_heading(latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float) -> float:
    """Find the heading of an aircraft moving from point A (latitude_1, longitude_1) to point B (latitude_2, longitude_2).
    Assumes the Earth is a spheroid - reasonable.
    
    Source:
        https://www.movable-type.co.uk/scripts/latlong.html
    """
    
    longitude_difference = longitude_2 - longitude_1
    return atan2(sin(longitude_difference) * cos(latitude_2), cos(latitude_1) * sin(latitude_2) - sin(latitude_1) * cos(latitude_2) * cos(longitude_difference))

def heading_to_vector(heading: float, magnitude: float = 1.0) -> np.ndarray:
    """Converts a heading (radians) to a unit vector. Optional argument of vector magnitude.
    Heading is N=0, (+) direction = clockwise [i.e. aircraft_heading = 10° means 80° in polar where pos X = 0 and (+) direction = counterclockwise]."""

    # note that the vertical (y axis) component corresponds to cos(heading)
    return np.array([magnitude * sin(heading),
                        magnitude * cos(heading)])

def magnitude(vector: np.ndarray) -> float:
    """Calculate magnitude of numpy vector."""
    return np.sqrt(np.dot(vector, vector))

def convert_to_range_zero_to_two_pi(angle: float, epsilon=1e-9) -> float:
        """Bring an angle (radians) into the range [0, 2pi)."""
        
        while angle > 2 * pi:
            angle -= 2 * pi
        while angle < 0:
            angle += 2 * pi
            
        if abs(angle - 2 * pi) < epsilon:
            return 0
    
        return angle