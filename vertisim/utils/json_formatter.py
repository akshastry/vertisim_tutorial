# from  import Path
from copy import deepcopy
import pandas as pd
import json


def create_collection_template():
    collection_temp = json.loads(
        """
    {
    "type": "FeatureCollection",
    "features": [
    ]
    }
    """)
    return collection_temp


def create_agent_template():
    agent_temp = json.loads("""
    {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "LineString",
        "coordinates": []
    }
    }
    """)
    return agent_temp


def create_trajectory_json(df: pd.DataFrame, agent_type: str, only_aircraft_simulation: False) -> dict:
    """
    Function that creates a geoJSON file from a pandas dataframe
    :param df: 
    :param agent_type: 'aircraft' or 'passenger'
    :return: 
    """
    collection = create_collection_template()

    if only_aircraft_simulation:
        return None

    if agent_type == 'aircraft':
        df1_grouped = df.groupby(by='tail_number')
    elif agent_type == 'passenger':
        df1_grouped = df.groupby(by='passenger_id')
    else:
        raise ValueError('The agent_type is not supported. It should be either aircraft or passenger.')

    for group_name, df_group in df1_grouped:
        agent = create_agent_template()
        agent['properties']['id'] = group_name
        df_filter = df_group[['longitude', 'latitude', 'altitude', 'time']]
        agent['geometry']['coordinates'] = df_filter.values.tolist()
        # agent['properties']['event'] = df_group['event'].tolist()
        # agent['properties']['location'] = df_group['location'].tolist()
        # agent['properties']['speed'] = df_group['speed'].tolist()
        # agent['properties']['soc'] = df_group['soc'].tolist()
        # agent['properties']['priority'] = df_group['priority'].tolist()
        collection["features"].append(agent)
    return collection
