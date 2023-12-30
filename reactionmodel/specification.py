from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Model
from reactionmodel.option import InitialCondition, Configuration
from reactionmodel.parameters import Parametrization
from reactionmodel.util import FrozenDictionary, ImmutableArray

@dataclass(eq=False)
class SimulationSpecification():
    model: Model
    parameters: dict
    initial_condition: dict
    simulation_options: dict

    def get_frozen(self):
        return FrozenSimulationSpecification(
            self.model,
            Parametrization.make(self.parameters),
            InitialCondition.make(self.initial_condition),
            Configuration.make(self.simulation_options)
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, SimulationSpecification):
            return False
        return self.get_frozen() == __value.get_frozen()

@dataclass(frozen=True)
class FrozenSimulationSpecification():
    model: Model
    parameters: FrozenDictionary
    initial_condition: ImmutableArray
    simulation_options: FrozenDictionary