from vertisim.vertisim import VertiSim
import simpy

class InstanceManager:
    def __init__(self, config):
        self.config = config
        self.sim_instance = VertiSim(
            env=simpy.Environment(), 
            config=self.config
        )
        self.status = True
    
    def reset(self):
        self.status = False
        # self.sim_instance.reset()
        self.sim_instance = VertiSim(
            env=simpy.Environment(), 
            config=self.config 
        )
        self.status = self.sim_instance.status
    
    def get_initial_state(self):
        initial_state = self.sim_instance.get_initial_state()
        action_mask = self.sim_instance.action_mask(initial_state=True)
        return {"initial_state": initial_state, "action_mask": action_mask}
    
    def step(self, actions):
        if self.config["sim_mode"]["client_server"]:
            return self.sim_instance.step(actions)
        else:
            response = self.sim_instance.step(actions)
            return {
                "new_state": response[0],
                "reward": response[1],
                "terminated": response[2],
                "truncated": response[3],
                "action_mask": response[4]
            }
    
    def get_performance_metrics(self):
        return self.sim_instance.get_performance_metrics()
