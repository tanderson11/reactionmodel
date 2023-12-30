from dataclasses import dataclass
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

    def get_frozen(self, parameter_class=None, configuration_class=None):
        parametrization = parameter_class(**self.parameters) if parameter_class is not None else Parametrization.make(self.parameters)
        configuration   = configuration_class(**self.parameters) if configuration_class is not None else Configuration.make(self.simulation_options)
        return FrozenSimulationSpecification(
            self.model,
            parametrization,
            InitialCondition.make(self.initial_condition),
            configuration,
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