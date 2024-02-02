# from .utils.process_wind_data import WindDataProcessor
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.ticker import MultipleLocator
# import numpy as np
# import pandas as pd
# from .utils.vectorMath import heading_to_vector, lat_long_to_heading
# import matplotlib.colors as mcolors


# # --- MEAN, MIN, MAX, STD for each day of SF Bay Area Wind Data
# # 09/01/2022 - 08/22/2023, 366 days, September to August
# wdp = WindDataProcessor('vertisim/input/airspace/SJC_SFO_wind_data.csv')
# data = wdp.wind_data
# data['date'] = data['datetime'].dt.date

# # Separate the data for SFO and MV
# sfo_data = data[data['locationname'] == 'SFO'].copy()
# sjc_data = data[data['locationname'] == 'SJC'].copy()

# SFO_lat_long = (37.620421335323286, -122.39588448702193)
# SJC_lat_long = (37.35888488813405, -121.93134583170949)
# SFO_SJC_heading = lat_long_to_heading(SFO_lat_long[0], SFO_lat_long[1], SJC_lat_long[0], SJC_lat_long[1])
# SJC_SFO_heading = lat_long_to_heading(SJC_lat_long[0], SJC_lat_long[1], SFO_lat_long[0], SFO_lat_long[1])
# SFO_SJC_route_vector = heading_to_vector(SFO_SJC_heading)
# SJC_SFO_route_vector = heading_to_vector(SJC_SFO_heading)

# # Function to plot wind data for a given location
# def plot_wind_speeds(location_data, location_name):
#     daily_stats = location_data.groupby('date')['windspeed'].agg(['mean', 'min', 'max', 'std'])
#     dates = [date for date, stats in daily_stats.iterrows()] 
#     mean_wind_speeds = daily_stats['mean'].round(2).tolist()
#     max_wind_speeds = daily_stats['max'].round(2).tolist()

#     plt.figure()
#     plt.rcParams['font.family'] = 'Serif'
#     plt.title(f"Wind in {location_name} Area, 09/01/2022 - 08/22/2023", pad=15)
#     plt.xlabel('Date', labelpad=15)
#     plt.ylabel('Wind Speed (m/s)', labelpad=15)
#     plt.plot(dates, mean_wind_speeds, color='darkblue', label='Mean Wind Speed')
#     plt.plot(dates, max_wind_speeds, color='lightblue', label='Max Wind Speed')

#     # thresholds
#     plt.axhline(y=12.86, color='#CC0000', linestyle='-', label='Cancellation Threshold')
#     plt.axhline(y=10.29, color='#FF6666', linestyle='-', label='Separation II Threshold')
#     plt.axhline(y=8.75, color='#FFCCCC', linestyle='-', label='Separation I Threshold')

#     # x-axis date format
#     locator = mdates.MonthLocator()
#     fmt = mdates.DateFormatter('%Y-%m-%d')
#     X = plt.gca().xaxis
#     X.set_major_locator(locator)
#     X.set_major_formatter(fmt)
#     plt.xticks(rotation=45)

#     plt.tight_layout()
#     plt.legend(title='Wind Speed Categories')
#     plt.show()

# def plot_threshold_counts(location_data, location_name):
#     # Define bins, labels, and colors for wind speed categories
#     bins = [0, 8.75, 10.29, 12.86, float('inf')]
#     labels = ['Nominal', 'Separation I', 'Separation II', 'Cancellation']
#     colors = ['#D0E4F7', '#A9CCE3', '#5499C7', '#154360']  # Blues from light to dark
    
#     location_data = location_data.copy()
#     location_data['category'] = pd.cut(location_data['windspeed'], bins=bins, labels=labels, include_lowest=True)
    
#     # Group by date and category, then unstack
#     category_counts = location_data.groupby(['date', 'category']).size().unstack(fill_value=0)
#     category_counts.index = pd.to_datetime(category_counts.index)

#     # Prepare the figure and axes for plotting
#     fig, ax = plt.subplots(figsize=(14, 7))

#     # Plot each category count as a separate bar with corresponding color
#     bottoms = np.zeros(len(category_counts))
#     for i, category in enumerate(labels):
#         ax.bar(category_counts.index, category_counts[category], bottom=bottoms, label=category, color=colors[i])
#         bottoms += category_counts[category].fillna(0).values

#     # Configure plot aesthetics and labels
#     plt.rcParams['font.family'] = 'Serif'
#     ax.set_xlabel('Date', labelpad=15)
#     ax.set_ylabel('Hour Count', labelpad=15)
#     ax.set_title(f'Hours per Day Between Wind Speed Categories at {location_name}, 09/01/2022 - 08/22/2023', pad=15)
    
#     # Set x-tick labels for the first of every month
#     first_of_month = pd.date_range(start=category_counts.index.min(), 
#                                    end=category_counts.index.max(), 
#                                    freq='MS')
#     ax.set_xticks(first_of_month)
#     ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in first_of_month], rotation=45)
    
#     # Set the y-axis major tick frequency
#     ax.yaxis.set_major_locator(MultipleLocator(1 if max(bottoms) < 25 else 2))

#     # Add light grey grid
#     ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='grey')

#     # Ensure tight layout and display legend
#     plt.legend(title='Wind Speed Categories')
#     plt.tight_layout()
#     plt.show()

# def plot_wind_directions(location_data, location_name, route_name, route_heading_vec):
#     """X axis: days. Y axis: sum of wind_dir_vec dot heading_vec for each day."""
#     daily_stats = location_data.groupby('date')
#     dates = []
#     cum_wind_dir = []
    
#     light_blue_color = '#ADD8E6'
#     blue_color = '#0000FF'
#     dark_blue_color = '#00008B'

#     # custom color map
#     cmap = plt.cm.Blues 
    
#     for date, group in daily_stats:
#         sum_wind_dir = 0
#         for wind_dir in group['winddir']:
#             wind_dir_vec = heading_to_vector(wind_dir)
#             sum_wind_dir += np.dot(wind_dir_vec, route_heading_vec)
#         dates.append(date)
#         cum_wind_dir.append(sum_wind_dir)
        
#     plt.figure()
#     plt.rcParams['font.family'] = 'Serif'
#     plt.title(f"Cumulative Wind Direction in {location_name} Area, Route {route_name}", pad=15)
#     plt.xlabel('Date', labelpad=15)
#     plt.ylabel('Cumulative Wind Direction Alignment', labelpad=15)
#     norm = mcolors.Normalize(vmin=-50, vmax=24)
#     plt.scatter(dates, cum_wind_dir, c=cum_wind_dir, cmap=cmap, norm=norm, label='Cumulative Wind Direction Alignment')

#     # thresholds
#     plt.axhline(y=24, color=cmap(norm(24)), linestyle='-', label='Pure Tailwind')
#     plt.axhline(y=0, color=cmap(norm(0)), linestyle='-', label='Pure Crosswind')
#     plt.axhline(y=-24, color= cmap(norm(-24)), linestyle='-', label='Pure Headwind')

#     # x-axis date format
#     locator = mdates.MonthLocator()
#     fmt = mdates.DateFormatter('%Y-%m-%d')
#     X = plt.gca().xaxis
#     X.set_major_locator(locator)
#     X.set_major_formatter(fmt)
#     plt.xticks(rotation=45)

#     plt.tight_layout()
#     plt.legend(title='Wind Direction Categories')
#     plt.show()

# plot_wind_speeds(sfo_data, 'SFO')
# plot_threshold_counts(sfo_data, 'SFO')
# plot_wind_speeds(sjc_data, 'SJC')
# plot_threshold_counts(sjc_data, 'SJC')
# plot_wind_directions(sfo_data, 'SFO', 'SFO to SJC', SFO_SJC_route_vector)
# plot_wind_directions(sfo_data, 'SFO', 'SJC to SFO', SJC_SFO_route_vector)
# plot_wind_directions(sjc_data, 'SJC', 'SFO to SJC', SFO_SJC_route_vector)
# plot_wind_directions(sjc_data, 'SJC', 'SJC to SFO', SJC_SFO_route_vector)
