from utils.distribution_generator import DistributionGenerator
from typing import Dict, List

def create_aircraft_arrival_process(arrival_process: Dict) -> List:
    distribution = DistributionGenerator(distribution_params=arrival_process['aircraft_arrival_distribution'],
                                    max_val_in_dist=arrival_process['aircraft_arrival_distribution']['max_val_in_dist'])
    return [distribution.pick_number_from_distribution() for _ in range(arrival_process['num_aircraft'])]