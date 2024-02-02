class PhaseParameters:

    def __init__(self, *parameters):
        self.signature = tuple(parameters)

    def __eq__(self, other):
        if isinstance(other, PhaseParameters):
            return self.signature == other.signature
        return False

    def __hash__(self) -> int:
        return hash(self.signature)
