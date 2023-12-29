from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Model

@dataclass(frozen=True)
class SimulationSpecification():
    model: Model
    parameters: dataclass
    initial_condition: np.ndarray
    simulation_options: dict