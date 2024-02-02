from .base_simulator import BaseSimulator

class StepSimulator(BaseSimulator):
    def __init__(self, sim_setup, env):
        super().__init__(sim_setup, env)

class DenseStopSimulator(StepSimulator):
    def run(self):
        # Implement the dense stop logic here
        pass

class SparseStopSimulator(StepSimulator):
    def run(self):
        # Implement the sparse stop logic here
        pass