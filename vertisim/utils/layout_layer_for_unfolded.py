from shapely.geometry import Point, Polygon
import numpy as np
import geog
import json
import pandas as pd
import os
from pathlib import Path


def circle(row, object_name):
    row['alt_m'] = 0
    p = Point([row[f'{object_name}_lon'], row[f'{object_name}_lat'], row['alt_m']])
    n_points = 20
    diameter = row[f'{object_name}_diameter_ft']
    d_m = diameter
    angles = np.linspace(0, 360, n_points)
    coords = np.array([p.x, p.y, p.z])  # Extract the coordinates from the Point object
    polygon = geog.propagate(coords, angles, d_m)  # Pass the coordinates instead of the Point object
    # polygon = geog.propagate(p, angles, d_m)
    return Polygon(polygon)


def circle_column(df, object_name):
    return df.apply(lambda row: circle(row, object_name), axis=1)


def create_vertiport_layout_layer_for_unfolded(df1, df2, df3, height, fato='fato', park='parking_pad', inters='inters'):
    df1 = circle_column(df=df1, object_name=fato)
    df2 = circle_column(df=df2, object_name=park)
    df3 = circle_column(df=df3, object_name=inters)

    # Store elevation data in the DataFrames
    # df1['elevation'] = height
    # df2['elevation'] = height
    # df3['elevation'] = height 

    concatenated_frame = pd.concat([df1, df2, df3]).reset_index()
    concatenated_frame.rename(columns={"index": "object", 0: "polygon"}, inplace=True)
    concatenated_frame['elevation'] = height
    return concatenated_frame


def dataframe_to_geojson(df):
    geojson_dict = {
        "type": "FeatureCollection",
        "features": []
    }

    for index, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "object": row["object"],
                "elevation": row["elevation"]
            },
            "geometry": row["polygon"].__geo_interface__
        }
        geojson_dict["features"].append(feature)
    
    return geojson_dict


def save_vertiport_layout_layer_for_unfolded(df1, df2, df3, height, layout_type, output_folder_path):
    concatenated_frame = create_vertiport_layout_layer_for_unfolded(df1, df2, df3, height)
    geojson_dict = dataframe_to_geojson(concatenated_frame)
    path_obj = Path(output_folder_path)
    # Navigate two levels up
    parent_dir = path_obj.parent.parent
    # Get the string representation of the parent directory
    parent_dir_str = str(parent_dir)
    output_folder_path = f'{parent_dir_str}/output'
    os.makedirs(output_folder_path, exist_ok=True)
    layout_path = f'{parent_dir_str}/{layout_type}.json'
    file_path = os.path.join(output_folder_path, f'{layout_type}.json')
    with open(file_path, 'w') as f:
        json.dump(geojson_dict, f)
