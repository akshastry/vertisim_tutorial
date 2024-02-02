from enum import Enum, auto
from pydantic import BaseModel
from typing import List, Type, Tuple

def generate_action_enum(vertiport_count: int) -> Type[Enum]:
    """
    Dynamically generates an ActionEnum class with members based on the provided vertiport count.
    """
    members = {
        'DONOTHING': auto(),
        'IDLE2CHARGE': auto()
    }
    
    for i in range(vertiport_count):
        members[f'IDLE2SERVICE_{i}'] = auto()
    
    return Enum('ActionEnum', members)

def create_action_models(vertiport_count: int) -> Tuple[Type[Enum], BaseModel]:
    """
    Dynamically creates ActionEnum and Actions models based on the provided vertiport count.
    """
    ActionEnum = generate_action_enum(vertiport_count)
    
    class Actions(BaseModel):
        actions_list: List[ActionEnum]

    return Actions

class SimulationResponse(BaseModel):
    new_state: dict
    reward: float
    terminated: bool
    truncated: dict





