from .base_simulator import BaseSimulator

class UninterruptedSimulator(BaseSimulator):
    def __init__(self, env, sim_setup):
        super().__init__(env, sim_setup)

    def run(self):
        if self.sim_setup.sim_mode['offline_optimization']:
            self.env.run()
        else:
            self.env.run(until=self.sim_setup.sim_params['max_sim_time'])
