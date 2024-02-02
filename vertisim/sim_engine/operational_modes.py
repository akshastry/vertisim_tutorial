

class OperationalMode:
    def __init__(self, simulator):
        self.simulator = simulator

    def execute(self):
        raise NotImplementedError

class OnDemand(OperationalMode):
    def execute(self):
        # Execute on-demand logic
        self.simulator.run()

class OfflineOptimization(OperationalMode):
    def execute(self):
        # Execute offline optimization logic
        self.simulator.run()

class ClientServer(OperationalMode):
    def execute(self):
        # Execute client-server logic. This only supports step-by-step simulation.
        self.simulator.run()