from typing import Dict
from ..aircraft.aircraft import AircraftStatus
from ..utils.helpers import current_and_lookahead_pax_count
import numpy as np


class ActionMask:
    """
    Class for creating action masks for the RL agent.
    """
    def __init__(self, config: Dict, sim_setup: object) -> None:
        self.config = config
        self.sim_setup = sim_setup

    def get_action_mask(self):
        """
        Create the action mask for the current state of the simulation.
        [Fly, Charge, Do nothing]
        """
        mask = []
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            # Can do any action
            if aircraft.status == AircraftStatus.IDLE \
                and aircraft.soc <= 100-self.config['external_optimization_params']['soc_increment_per_charge_event'] \
                    and aircraft.soc >= self.config['aircraft_params']['min_reserve_soc']:
                destination_mask = [1 if destination != aircraft.current_vertiport_id 
                                    else 0 for destination in self.sim_setup.vertiport_ids]
                mask.extend(destination_mask + [1, 1])
            # Can only charge
            elif aircraft.status == AircraftStatus.IDLE \
                    and aircraft.soc <= self.config['aircraft_params']['min_reserve_soc']:
                destination_mask = [0] * self.sim_setup.num_vertiports()
                mask.extend(destination_mask + [1, 0])
            # Can fly and do nothing
            elif aircraft.status == AircraftStatus.IDLE and aircraft.soc > 100-self.config['external_optimization_params']['soc_increment_per_charge_event']:
                destination_mask = [1 if destination != aircraft.current_vertiport_id 
                                    else 0 for destination in self.sim_setup.vertiport_ids] 
                mask.extend(destination_mask + [0, 1])               
            elif aircraft.status == AircraftStatus.CHARGE or aircraft.status == AircraftStatus.FLY:
                destination_mask = [0] * self.sim_setup.num_vertiports()
                mask.extend(destination_mask + [0, 1])               
            
        # Review the mask based on demand. If there is no current and future demand at any particular vertiport, mask out the action
        for vertiport_id, vertiport in self.sim_setup.vertiports.items():
            if current_and_lookahead_pax_count(vertiport) == 0:
                # Find the index of the vertiport in the mask for each aircraft and mask out the action
                vertiport_index = self.sim_setup.vertiport_id_to_index_map[vertiport_id]
                # Iterate through each aircraft's part of the mask
                for i in range(0, len(mask), self.sim_setup.num_vertiports() + 2):
                    mask[i + vertiport_index] = 0                        
                    
        # Flatten the mask
        return np.array(mask).flatten().tolist()