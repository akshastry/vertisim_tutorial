import sqlite3
import datetime
from collections import defaultdict
from json import dumps
import json
from typing import Dict, List
import pandas as pd
import os
from .utils.json_formatter import create_trajectory_json
from .utils.helpers import get_str_before_first_occurrence_of_char

def create_connection(db_file: str):
    try:
        with sqlite3.connect(db_file) as conn:
            return conn
    except sqlite3.Error as e:
        print(e)
        return None
    
def save_output(simulation_params: dict, 
                trajectories: dict,
                performance_metrics: defaultdict, 
                simulationID: str, 
                flight_directions: list) -> str:

    # Save trajectories
    if simulation_params['sim_params']['save_trajectories']:
        save_trajectories(trajectories=trajectories,
                          simulation_params=simulation_params,
                          output_dir=simulation_params['output_params']['output_folder_path'])                
    
    # Save performance metrics
    save_performance_metrics(simulation_params=simulation_params,
                                performance_metrics=performance_metrics,
                                simulationID=simulationID,
                                flight_directions=flight_directions)
    
def save_performance_metrics(simulation_params: dict,
                             performance_metrics: defaultdict,
                             simulationID: str,
                             flight_directions: list) -> str:

    output_path = f"{simulation_params['output_params']['output_folder_path']}/sqlite/db/vertisimDatabase.sqlite"
    print(f"Saving performance metrics to {output_path}\n")
    # If path doesn't exist, create it
    if not os.path.exists(f"{simulation_params['output_params']['output_folder_path']}/sqlite/db"):
        os.makedirs(f"{simulation_params['output_params']['output_folder_path']}/sqlite/db", exist_ok=True)

    conn = create_connection(output_path)
    assert conn, 'Could not establish database connection.'
    cursor = conn.cursor()
    
    # --- SIMULATION TABLE ---
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config = dumps(simulation_params)
        
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Simulation (
                SimulationID VARCHAR(8) PRIMARY KEY,
                Config TEXT,
                Timestamp DATETIME
            );
        """)
        
        cursor.execute("SELECT SimulationID FROM Simulation WHERE SimulationID = ?", (simulationID,))
        data=cursor.fetchone()
        if data is None:
            cursor.execute("INSERT INTO Simulation (SimulationID, Config, Timestamp) VALUES (?, ?, ?)", (simulationID, config, timestamp))
        else:
            cursor.execute("UPDATE Simulation SET Timestamp = ?, Config = ? WHERE SimulationID = ?", (timestamp, config, simulationID))
            
    except sqlite3.Error as err:
        print(f"Error: {err}")
    
    # --- PERFORMANCE METRICS TABLE --- 
    metric_names = ['EnergyConsumptionMean', 'EnergyConsumptionMedian', 'EnergyConsumptionStd', 'EnergyConsumptionMax', 'EnergyConsumptionMin', 'EnergyConsumptionTotal', 'FlightDurationMean', 'FlightDurationMedian', 'FlightDurationStd', 'FlightDurationMax', 'FlightDurationMin', 'TripTimeMean', 'TripTimeMedian', 'TripTimeStd', 'TripTimeMax', 'TripTimeMin']

    try:
        # Create the table with dynamic columns
        columns = ', '.join([f"{metric} FLOAT" for metric in metric_names])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS PerformanceMetrics (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            SimulationID VARCHAR(8),
            FlightDirection TEXT,
            {columns},
            FOREIGN KEY (SimulationID) REFERENCES Simulation(SimulationID));
        """)
    except sqlite3.Error as err:
        print(f"Error: {err}")
        
    try:
        for flight_dir in flight_directions:
            metrics_for_flight = performance_metrics[flight_dir]
            energy_metrics = metrics_for_flight['energy_consumption']
            flight_duration_metrics = metrics_for_flight['flight_duration']
            trip_time_metrics = metrics_for_flight['passenger_trip_time']
            metric_values = [
                energy_metrics['mean'], 
                energy_metrics['median'], 
                energy_metrics['std'],
                energy_metrics['max'], 
                energy_metrics['min'], 
                energy_metrics['total'],
                flight_duration_metrics['mean'], 
                flight_duration_metrics['median'], 
                flight_duration_metrics['std'],
                flight_duration_metrics['max'], 
                flight_duration_metrics['min'],
                trip_time_metrics['mean'],
                trip_time_metrics['median'],
                trip_time_metrics['std'],
                trip_time_metrics['max'],
                trip_time_metrics['min']
            ]
    
            placeholder = ', '.join(['?'] * (len(metric_values) + 2))
            sql_insert_query = f"INSERT INTO PerformanceMetrics (SimulationID, FlightDirection, {', '.join(metric_names)}) VALUES ({placeholder});"
            cursor.execute(sql_insert_query, [simulationID, flight_dir] + metric_values)
            
    except sqlite3.Error as err:
        print(f"Error: {err}")

    conn.commit()
    cursor.close()
    conn.close()
    
    return os.getcwd() + output_path


def save_trajectories(trajectories: Dict,
                      simulation_params: Dict,
                      output_dir: str) -> None:
    # Save agent trajectories
    for key, value in trajectories.items():
        df = convert_dict_to_dataframe(dic=value, orient='index')
        trajectory_output_file_name = get_output_filename(output_dir=output_dir, file_name=key)
        save_df_to_csv(df=df, output_file_name=trajectory_output_file_name)

        # Convert trajectory df to geoJSON and save
        agent_type = get_str_before_first_occurrence_of_char(key, '_')
        if json_trajectory := create_trajectory_json(
                df=df, agent_type=agent_type,
                only_aircraft_simulation=simulation_params['sim_params']['only_aircraft_simulation']
        ):
            with open(trajectory_output_file_name + '.geojson', 'w') as f:
                json.dump(json_trajectory, f)

def convert_dict_to_dataframe(dic: Dict, orient: str = 'columns') -> pd.DataFrame:
    df = pd.DataFrame.from_dict(dic, orient=orient)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    return df

def get_output_filename(output_dir: str, file_name: str) -> str:
    return os.path.join(output_dir, file_name)

def save_df_to_csv(df: pd.DataFrame, output_file_name: str) -> None:
    df.to_csv(f'{output_file_name}.csv')