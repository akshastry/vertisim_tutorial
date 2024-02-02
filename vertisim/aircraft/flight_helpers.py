import numpy as np

G_CONSTANT = 9.80665  # m/s^2

def temperature(altitude: float) -> float:
    """
    Computes the temperature at a given altitude
    :param altitude: in m
    :return: temperature in K
    """
    return 288.16 - 6.5*10**-3 * altitude

def lift_induced_drag_coef(cd_0: float, ld_max: float) -> float:
    """
    From the paper:
    The promise of energy-efficient battery-powered urban aircraft
    """
    return 1 / (4*cd_0*ld_max**2)

def climb_power_consumption_for_lift(tom, vertical_velocity):
    """
    From the paper:
    The promise of energy-efficient battery-powered urban aircraft   

    tom: take-off mass in kg 
    """    
    return weight(tom)*vertical_velocity

def climb_power_consumption_for_drag(altitude, atmosphere_condition, wing_area, cd_0, ld_max, tom, horizontal_speed):
    """
    From the paper:
    The promise of energy-efficient battery-powered urban aircraft
    """
    term2 = 1/2 * rho(altitude, atmosphere_condition) * wing_area * cd_0 * horizontal_speed ** 3
    term3 = lift_induced_drag_coef(cd_0, ld_max) * weight(tom)**2 / (1/2*rho(altitude, atmosphere_condition) \
                                    * wing_area * horizontal_speed)   
    return term2 + term3

def weight(mass):
    """
    Computes the weight of an aircraft given its mass
    :param mass: in kg
    :return: weight in N
    """
    return round(mass * G_CONSTANT)

def rotor_disk_area(mtom, disk_load):
    """
    Computes the rotor disk area given the maximum take-off mass and the disk load
    :param mtom: in kg
    :param disk_load: in kg/m^2
    :return: rotor disk area in m^2
    """
    return mtom / disk_load

def atmosphere_params(condition: str):
    if condition == 'good':
        tgl = 288.15 # Ground temperature [K]
        dgl = 1.225 # Ground density [kg/m^3]
        return tgl, dgl
    elif condition == 'bad':
        tgl = 300 # Ground temperature [K]
        dgl = 0.974 # Ground density [kg/m^3]
        return tgl, dgl
    else:
        raise ValueError('Invalid atmosphere condition. Choose between "good" and "bad"')

def rho(altitude: float, atmosphere_condition: str='good'):
    """
    Computes the air density at a given altitude
    :param altitude: in m
    :return: air density in kg/m^3
    """
    tgl, dgl = atmosphere_params(atmosphere_condition)
    return round(dgl * (temperature(altitude)/tgl)**((G_CONSTANT/(287*6.5*10**-3))-1), 4)

def stall_speed(atmosphere_condition, altitude, mtom, wing_area, cl_max):
    """
    Computes the stall speed of an aircraft given its mass
    :param mass: in kg
    :return: stall speed in m/s
    """
    return round(np.sqrt((2*weight(mtom))/(rho(altitude, atmosphere_condition)*wing_area*cl_max)))

