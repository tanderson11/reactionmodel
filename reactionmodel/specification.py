from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Model

@dataclass(frozen=True)
class SimulationSpecification():
    model: Model = None
    parameters: dict = {}
    initial_condition: np.ndarray = np.array([])
    simulation_options: dict = {}