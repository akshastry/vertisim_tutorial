import sqlite3
import pandas as pd


def fetch_data_from_db(db_path):
    """
    Connects to the SQLite database file and fetches the Timestamp and SFO_MV_TRIP_TIME_MEAN columns.
    If data is found, it returns a dictionary with the data; otherwise, it returns a dictionary with the status 'No Data'.

    :param db_path: Path to the SQLite database file
    :return: Dictionary with the data or status
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        # Formulate the SQL query to join the tables on 'SimulationID' and select the required columns
        query = """
        SELECT s.Timestamp, p.SFO_MV_TRIP_TIME_MEAN
        FROM Simulation s
        JOIN PerformanceMetrics p ON s.SimulationID = p.SimulationID;
        """
        # Execute the query and fetch the data
        data = pd.read_sql_query(query, conn)
        # Close the connection to the database
        conn.close()

        # Check if the dataframe is empty
        if data.empty:
            return {'status': 'No Data'}
        
        # If data is found, convert the dataframe to a dictionary
        return data.to_dict(orient='list')
    
    except sqlite3.Error as e:
        # If an error occurs, return the error message
        return {'status': 'Failed', 'error': str(e)}
    

def fetch_latest_data_from_db(db_path):
    """
    Connects to the SQLite database file and fetches the latest Timestamp and SFO_MV_TRIP_TIME_MEAN columns.
    If data is found, it returns a dictionary with the data; otherwise, it returns a dictionary with the status 'No Data'.

    :param db_path: Path to the SQLite database file
    :return: Dictionary with the latest data or status
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Formulate the SQL query to join the tables on 'SimulationID' and select the required columns
        # Orders by Timestamp descending and limits the result to the latest entry
        query = """
        SELECT s.Timestamp, p.SFO_MV_TRIP_TIME_MEAN
        FROM Simulation s
        JOIN PerformanceMetrics p ON s.SimulationID = p.SimulationID
        ORDER BY s.Timestamp DESC
        LIMIT 1;
        """
        # Execute the query and fetch the data
        data = pd.read_sql_query(query, conn)
        # Close the connection to the database
        conn.close()

        # Check if the dataframe is empty
        if data.empty:
            return {'status': 'No Data'}
        
        # If data is found, convert the dataframe to a dictionary
        return data.to_dict(orient='records')[0]
    
    except sqlite3.Error as e:
        # If an error occurs, return the error message
        return {'status': f'Database is not ready. Wait...', 'error': str(e), }

