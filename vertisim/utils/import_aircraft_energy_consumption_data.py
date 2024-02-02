from .read_files import read_input_file


def import_aircraft_energy_consumption_data(folder_path: str, aircraft_models: list, ranges: list) -> dict:
    """Import aircraft energy consumption data from the data folder.

    Returns:
        aircraft_energy_consumption_data (Dict): Aircraft energy consumption data.

    """
    aircraft_energy_consumption_database = {}
    for aircraft_model in aircraft_models:
        # Get file path. If the file does not exist, raise error.
        try:
            file_path = f'{folder_path}/{aircraft_model}.xlsx'
        except ValueError as e:
            raise ValueError(f'No energy consumption data for {aircraft_model} is found.') from e

        df = read_input_file(file_path)
        df = df[df.range.isin(ranges)]  # This tries to match exactly the same ranges that are calculated from
        # vert-vert distances. If they are not matching exactly, they will be excluded.
        aircraft_energy_consumption_database[aircraft_model] = df
    return aircraft_energy_consumption_database
