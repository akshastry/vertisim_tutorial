import json
import subprocess
from math import pi
import numpy as np
import os
from tqdm import tqdm

config_file_path = './config_supernal.json'
runs = 10
# Load the existing config.json into a Python dictionary
with open(config_file_path, 'r') as f:
    config = json.load(f)
    
# Write the updated config back to config.json
with open(config_file_path, 'w') as f:
    json.dump(config, f, indent=4)

wind_magnitudes = [0, 10, 20, 30, 40, 50]
wind_angles = np.arange(0, 361, 45)  # Angles in degrees

total_iterations = runs * len(wind_magnitudes) * len(wind_angles)
progress_bar = tqdm(total=total_iterations)

# Loop through each combination of wind_magnitude and wind_angle
for _ in range(runs):
    for magnitude in wind_magnitudes:
        for angle in wind_angles:
            # print(f"Running simulation with wind_magnitude={magnitude} and wind_angle={angle} degrees.")
            
            # Update the config dictionary with new values
            config['airspace_params']['wind_magnitude_mph'] = int(magnitude)
            config['airspace_params']['wind_angle_degrees'] = int(angle)

            # Write the updated config back to config.json
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Run the vertisim.runner module
            result = subprocess.run(['python', '-m', 'vertisim.vertisim.runner', '--config', config_file_path])
            if result.returncode != 0:
                print(f"An error occurred when running the simulation with wind_magnitude={magnitude} and wind_angle={angle}. Return code: {result.returncode}")

            # Update the progress bar after each iteration
            progress_bar.update(1)

# Close the progress bar when done
progress_bar.close()