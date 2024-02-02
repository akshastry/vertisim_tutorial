from typing import Dict

def get_aircraft_count(network_and_demand_params: Dict):
    """
    Get the number of aircraft at the initialization of the simulation.

    Args:
    - network_and_demand_params: Dictionary containing network and demand parameters.

    Returns:
    - Number of aircraft in the simulation.
    """
    return sum(
        vertiport['aircraft_arrival_process'][
            'num_initial_aircraft_at_vertiport']
        for _, vertiport in network_and_demand_params['vertiports'].items()
    )
