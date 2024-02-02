class BaseSimulator:
    def __init__(self, env, sim_setup: object):
        self.env = env
        self.sim_setup = sim_setup

    def run(self):
        raise NotImplementedError("The run method should be implemented in derived classes")