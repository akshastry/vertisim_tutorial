import pandas as pd
import random
from typing import Dict, List
from ..utils.read_files import read_input_file
from ..utils.helpers import set_seed
from ..utils.units import mph_to_metersec, degrees_to_radians
from ..wind.wind_models import DynamicWindState
import datetime as dt
from math import pi


class WindDataProcessor:
    def __init__(self, file_path):
        self.wind_data = self._load_wind_data(file_path)
        self.wind_data_day = self.get_wind_data_for_random_day()

    @staticmethod
    def _load_wind_data(file_path):
        data = read_input_file(file_path)
        data = data[['locationname', 'datetime', 'windspeed', 'winddir']]  
        data = WindDataProcessor.exclude_days_with_missing_hours(data)
        return WindDataProcessor.convert_winddata_units(data)
    
    @staticmethod
    def exclude_days_with_missing_hours(wind_data):
        # Convert 'datetime' to datetime object if it's not already
        if not pd.api.types.is_datetime64_any_dtype(wind_data['datetime']):
            wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])
        
        # Group by date and location, and filter out groups with less than 24 records (hours)
        grouped = wind_data.groupby([wind_data['datetime'].dt.date, 'locationname'])
        full_days = grouped.filter(lambda x: len(x) == 24)

        # Get unique dates for each location
        full_days_dates = full_days.groupby('locationname')['datetime'].apply(lambda x: x.dt.date.unique())

        # Find intersection of dates (dates that are complete for all locations)
        complete_dates = set(full_days_dates.iloc[0])
        for dates in full_days_dates.iloc[1:]:
            complete_dates.intersection_update(dates)

        # Filter original data to exclude days with missing data
        mask = (wind_data['datetime'].dt.date.isin(complete_dates)) & (wind_data['locationname'].isin(full_days_dates.index))
        cleaned_data = wind_data[mask]

        return cleaned_data.reset_index(drop=True)
    
    @staticmethod
    def convert_winddata_units(wind_data):
        wind_data['windspeed'] = wind_data['windspeed'].apply(mph_to_metersec)
        wind_data['winddir'] = wind_data['winddir'].apply(degrees_to_radians)
        return wind_data


    def pick_random_day(self, max_wind_speed_threshold=None) -> dt.date:
        """
        Picks a random day from the wind data. Optionally, specifiy a threshold for the max wind speed on that day (at least max_wind_speed_threshold) [m/s].
        """
        set_seed(how='instant')
        if max_wind_speed_threshold:
            grouped_data = self.wind_data.groupby(self.wind_data['datetime'].dt.date)['windspeed'].max()
            filtered_days = grouped_data[grouped_data > max_wind_speed_threshold].index.tolist()
            if len(filtered_days) == 0:
                raise ValueError(f"No days with maximum windspeed greater than {max_wind_speed_threshold} m/s found.")
            return random.choice(filtered_days)
        else:
            unique_days = self.wind_data['datetime'].dt.date.unique().tolist()
            return random.choice(unique_days)

    def get_wind_data_for_random_day(self):
        """
        Filters the wind data for a randomly selected day.
        """
        random_day = self.pick_random_day()
        mask = (self.wind_data['datetime'].dt.date == random_day)
        return self.wind_data[mask].reset_index(drop=True)
    
    def _get_filtered_data(self, time_of_day=None, location=None):
        mask = True
        if time_of_day is not None:
            mask &= self.wind_data_day['datetime'].dt.hour == time_of_day
        if location is not None:
            mask &= self.wind_data_day['locationname'] == location
        return self.wind_data_day[mask]    

    def get_time_specific_wind_states(self, time_of_day: int) -> Dict:
        data = self._get_filtered_data(time_of_day=time_of_day)
        if data.empty:
            raise ValueError(f'No wind data for {time_of_day}:00.')
        return {loc: {'windspeed': ws, 'winddir': wd} for loc, ws, wd in zip(data['locationname'], data['windspeed'], data['winddir'])}

    def get_location_specific_wind_states(self, location: str) -> Dict:
        data = self._get_filtered_data(location=location)
        if data.empty:
            raise ValueError(f'No wind data for {location}.')
        return {time: {'windspeed': ws, 'winddir': wd} for time, ws, wd in zip(data['datetime'], data['windspeed'], data['winddir'])}

    def get_time_and_location_specific_wind_states(self, time_of_day: int, location: str, locations: List[str]) -> Dict:
        time_of_day %= 24
        data = self._get_filtered_data(time_of_day, location)
        if data.empty:
            raise ValueError(f'No wind data for {location} at {time_of_day}:00.')
        row = data.iloc[0]
        wind_state = DynamicWindState(locationname=locations.index(location), windspeed=row['windspeed'], winddir=row['winddir'])
        return wind_state.model_dump()