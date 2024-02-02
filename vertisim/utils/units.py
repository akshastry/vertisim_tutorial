import numpy as np
import pandas as pd
from math import pi


def sec_to_ms(sec):
    if sec is None:
        return None
    elif type(sec) in [np.ndarray, pd.core.series.Series]:
        return round(sec * 1000, 2)
    elif type(sec) == list:
        return [round(i * 1000, 2) for i in sec]
    elif type(sec) not in [int, float, np.float64, np.float32, np.int_, np.int32]:
        raise TypeError(f"Can't convert sec to ms. Duration (sec) format is not correct. The type you inputted is {type(sec)}")
    else:
        return round(sec * 1000, 2)
    
def min_to_ms(min):
    if min is None:
        return None
    elif type(min) in [np.ndarray, pd.core.series.Series]:
        return round(min * 60000, 2)
    elif type(min) == list:
        return [round(i * 60000, 2) for i in min]
    elif type(min) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert min to ms. Duration (min) format is not correct. The type you inputted is {type(min)}")
    else:
        return round(min * 60000, 2)


def min_to_sec(min):
    if min is None:
        return None
    elif type(min) in [np.ndarray, pd.core.series.Series]:
        return round(min * 60, 2)
    elif type(min) == list:
        return [round(i * 60, 2) for i in min]
    elif type(min) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert min to sec. Duration (min) format is not correct. The type you inputted is {type(min)}")
    else:
        return round(min * 60, 2)


def ms_to_min(ms):
    if ms is None:
        return None
    elif type(ms) in [np.ndarray, pd.core.series.Series]:
        return round(ms / 60000, 2)
    elif type(ms) == list:
        return [round(i / 60000, 2) for i in ms]
    elif type(ms) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert ms to min. Duration (ms) format is not correct. The type you inputted is {type(ms)}")
    else:
        return round(ms / 60000, 2)
    
def sec_to_min(sec):
    if sec is None:
        return None
    elif type(sec) in [np.ndarray, pd.core.series.Series]:
        return round(sec / 60, 2)
    elif type(sec) == list:
        return [round(i / 60, 2) for i in sec]
    elif type(sec) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert sec to min. Duration (sec) format is not correct. The type you inputted is {type(sec)}")
    else:
        return round(sec / 60, 2)
    
def ms_to_sec(ms):
    if ms is None:
        return None
    elif type(ms) in [np.ndarray, pd.core.series.Series]:
        return round(ms / 1000, 2)
    elif type(ms) == list:
        return [round(i / 1000, 2) for i in ms]
    elif type(ms) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert ms to sec. Duration (ms) format is not correct. The type you inputted is {type(ms)}")
    else:
        return round(ms / 1000, 2)
    
def hr_to_ms(hr):
    if hr is None:
        return None
    elif type(hr) in [np.ndarray, pd.core.series.Series]:
        return round(hr * 3600000, 2)
    elif type(hr) == list:
        return [round(i * 3600000, 2) for i in hr]
    elif type(hr) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert hr to ms. Duration (hr) format is not correct. The type you inputted is {type(hr)}")
    else:
        return round(hr * 3600000, 2)
    
def sec_to_hr(sec):
    if sec is None:
        return None
    elif type(sec) in [np.ndarray, pd.core.series.Series]:
        return round(sec / 3600, 2)
    elif type(sec) == list:
        return [round(i / 3600, 2) for i in sec]
    elif type(sec) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert sec to hr. Duration (sec) format is not correct. The type you inputted is {type(sec)}")
    else:
        return round(sec / 3600, 2)
    
def ms_to_hr(ms):
    if ms is None:
        return None
    elif type(ms) in [np.ndarray, pd.core.series.Series]:
        return round(ms / 3600000, 2)
    elif type(ms) == list:
        return [round(i / 3600000, 2) for i in ms]
    elif type(ms) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert ms to hr. Duration (ms) format is not correct. The type you inputted is {type(ms)}")
    else:
        return round(ms / 3600000, 2)

def ft_to_m(ft):
    if ft is None:
        return None
    elif type(ft) in [np.ndarray, pd.core.series.Series]:
        return round(ft * 0.3048, 2)
    elif type(ft) == list:
        return [round(i * 0.3048, 2) for i in ft]
    elif type(ft) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert ft to m. Altitude (ft) format is not correct. The type you inputted is {type(ft)}")
    else:
        return round(ft * 0.3048, 2)
    
def m_to_ft(m):
    if m is None:
        return None
    elif type(m) in [np.ndarray, pd.core.series.Series]:
        return round(m / 0.3048, 2)
    elif type(m) == list:
        return [round(i / 0.3048, 2) for i in m]
    elif type(m) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert m to ft. Altitude (m) format is not correct. The type you inputted is {type(m)}")
    else:
        return round(m / 0.3048, 2)

def miles_to_m(miles):
    if miles is None:
        return None
    elif type(miles) in [np.ndarray, pd.core.series.Series]:
        return round(miles * 1609.34, 2)
    elif type(miles) == list:
        return [round(i * 1609.34, 2) for i in miles]
    elif type(miles) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert miles to m. Distance (miles) format is not correct. The type you inputted is {type(miles)}")
    else:
        return round(miles * 1609.34, 2)
    
def mph_to_metersec(mph):
    if mph is None:
        return None
    elif type(mph) in [np.ndarray, pd.core.series.Series]:
        return round(mph * 0.44704, 2)
    elif type(mph) == list:
        return [round(i * 0.44704, 2) for i in mph]
    elif type(mph) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert mph to m/s. Speed (mph) format is not correct. The type you inputted is {type(mph)}")
    else:
        return round(mph * 0.44704, 2)
    
def metersec_to_mph(metersec):
    if metersec is None:
        return None
    elif type(metersec) in [np.ndarray, pd.core.series.Series]:
        return round(metersec / 0.44704, 2)
    elif type(metersec) == list:
        return [round(i / 0.44704, 2) for i in metersec]
    elif type(metersec) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert m/s to mph. Speed (m/s) format is not correct. The type you inputted is {type(metersec)}")
    else:
        return round(metersec / 0.44704, 2)
    

def watt_to_kw(watt):
    if watt is None:
        return None
    elif type(watt) in [np.ndarray, pd.core.series.Series]:
        return round(watt / 1000, 2)
    elif type(watt) == list:
        return [round(i / 1000, 2) for i in watt]
    elif type(watt) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert watt to kw. Power (watt) format is not correct. The type you inputted is {type(watt)}")
    else:
        return round(watt / 1000, 2)
    

def kw_to_watt(kw):
    if kw is None:
        return None
    elif type(kw) in [np.ndarray, pd.core.series.Series]:
        return round(kw * 1000, 2)
    elif type(kw) == list:
        return [round(i * 1000, 2) for i in kw]
    elif type(kw) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert kw to watt. Power (kw) format is not correct. The type you inputted is {type(kw)}")
    else:
        return round(kw * 1000, 2)


def degrees_to_radians(degrees):
    if degrees is None:
        return None
    if type(degrees) in [np.ndarray, pd.core.series.Series]:
        return round(degrees * 0.0174533, 4)
    elif type(degrees) == list:
        return [round(i * 0.0174533, 4) for i in degrees]
    elif type(degrees) not in [int, float, np.float64, np.float32, np.int_]:
        raise TypeError(f"Can't convert degrees to radians. Angle (degrees) format is not correct. The type you inputted is {type(degrees)}")
    return round(degrees * 0.0174533, 4)
    
    