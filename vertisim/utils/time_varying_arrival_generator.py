import numpy as np
import pandas as pd
from .helpers import set_seed

def get_arrival_rates(df):
    # Initialize an empty dictionary to hold the rates
    rate_dict = {}
    
    # Populate the dictionary
    for col in df.columns:
        if col != 'time_sec':
            origin, destination = col.split('_')
            if origin not in rate_dict:
                rate_dict[origin] = {}
            rate_dict[origin][destination] = list(df[col])
    
    time_intervals = list(df['time_sec'])
    return rate_dict, time_intervals


def generate_time_varying_arrival_times(arrival_rate_df, origin):
    # Get the arrival rates and time intervals
    rate_dict, time_intervals = get_arrival_rates(arrival_rate_df)

    destinations = list(rate_dict[origin].keys())
    results = []

    for destination in destinations:
        arrival_times = []
        previous_time = 0

        rates = rate_dict[origin][destination]

        for idx, rate in enumerate(rates):
            set_seed(how='instant')
            # Get the current and next time interval (or the end of the last interval)
            current_time = time_intervals[idx]
            next_time = time_intervals[idx + 1] if idx + 1 < len(time_intervals) else time_intervals[idx]

            # Calculate expected arrivals for this interval
            interval_duration = (next_time - current_time) / 3600  # convert to hours
            expected_arrivals = rate * interval_duration

            # Generate arrivals using Poisson distribution
            num_to_generate = np.random.poisson(expected_arrivals)

            for _ in range(num_to_generate):
                # Generate a random time within the current interval
                time_within_interval = np.random.uniform(current_time, next_time)

                # Append this time to the list of arrival times
                arrival_times.append({
                    'passenger_arrival_time': time_within_interval,
                    'origin_vertiport_id': origin,
                    'destination_vertiport_id': destination
                })

        # Add generated data for this destination to results
        results.extend(arrival_times)

    # Sort the list based on time
    results = sorted(results, key=lambda x: x['passenger_arrival_time'])

    return pd.DataFrame(results)