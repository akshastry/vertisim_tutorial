from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
import sys
sys.path.append('/app')

from vertisim.vertisim.instance_manager import InstanceManager
from .models import create_action_models
from .config import CONFIG  # Import the loaded configuration

class Actions(BaseModel):
    actions: List[int]
    # actions: int

app = FastAPI()

instance_manager = InstanceManager(config=CONFIG)

# Actions = create_action_models(instance_manager.sim_instance.get_vertiport_count())

@app.get("/status")
def get_status():
    if not instance_manager.status:
        raise HTTPException(status_code=503, detail="Vertisim is not ready.")
    return {"status": "Success", "message": "Vertisim is ready."}

@app.post("/reset")
def reset():
    instance_manager.reset()
    return {"status": "Success", "message": "Vertisim instance reset successfully."}

@app.get("/get_initial_state")
def get_initial_state():
    return instance_manager.get_initial_state()

@app.post("/step")
def step(actions: Actions):
    # print(f"actions from vertisim api: {actions.actions}")
    # TODO: Action input should be list of enums
    new_state, reward, terminated, truncated, action_mask = instance_manager.step(actions.actions)
    return {
        "new_state": new_state,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "action_mask": action_mask
    }

@app.get("/get_space_params")
def get_space_params():
    # Extract and return the parameters needed to construct action 
    # and observation spaces in the RL algorithm.
    return {
        "n_actions": instance_manager.sim_instance.get_action_count(),
        "n_aircraft": instance_manager.sim_instance.get_aircraft_count(), 
        "n_vertiports": instance_manager.sim_instance.get_vertiport_count(),
        "n_vertiport_state_variables": instance_manager.sim_instance.get_vertiport_state_variable_count(),
        "n_aircraft_state_variables": instance_manager.sim_instance.get_aircraft_state_variable_count(),
        "n_environmental_state_variables": instance_manager.sim_instance.get_environmental_state_variable_count(),
        "n_additional_state_variables": instance_manager.sim_instance.get_additional_state_variable_count()
    }

@app.get("/performance_metrics")
def get_performance_metrics():
    return instance_manager.sim_instance.get_performance_metrics()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001) 

