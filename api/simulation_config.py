from pydantic import BaseModel
import simpy

class SimulationConfig(BaseModel):
    env: simpy.Environment()
    config: dict
    reward_function_parameters: list
