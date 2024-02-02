import pandas as pd
import numpy as np
import json
from typing import Union
from .helpers import get_absolute_path

from typing import Union, Any

def read_input_file(file_path: str, sheet_name: str = 'Sheet1') -> Union[pd.DataFrame, Any]:
    """
    Function that can read the input file and create a pandas dataframe.
    The input file extension can be types of .csv, .parquet, .hdf5, and .xlsx
    :param sheet_name:
    :param file_path:
    :return: pd.DataFrame
    """
    file_type = detect_file_type(file_path)
    # file_path = get_absolute_path(file_path)
    if file_type is None:
        return None
    if file_type == 'csv':
        df = pd.read_csv(file_path)
        df = df.replace(np.nan, None)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_type == 'hdf5':
        df = pd.read_hdf(file_path)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    elif file_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f'The file type is not supported: {file_path}')

    # Filter out empty rows
    df = df.dropna(how='all')

    return df



def detect_file_type(file_path: str) -> str:
    """
    Function that detects the file type of the input file
    :param file_path:
    :return: str
    """
    return None if file_path is None else file_path.split('.')[-1]
