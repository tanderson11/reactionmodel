from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Model
from reactionmodel.util import FrozenDictionary

@dataclass(frozen=True)
class SimulationSpecification():
    model: Model
    parameters: FrozenDictionary
    initial_condition: np.ndarray
    simulation_options: FrozenDictionary