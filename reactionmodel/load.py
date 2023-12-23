import os
from dataclasses import dataclass
import numpy as np

from simulationspec import OptionParser, InitialConditionParser
from msl import ModelParser, ParameterParser
from model import Model

@dataclass(frozen=True)
class SimulationSpecification():
    model: Model
    parameters: dict
    options: dict
    initial_condition: np.ndarray

def load(path):
    model = ModelParser().load_model(os.path.join(path, 'model.txt'))
    parameters = ParameterParser().load_parameters(os.path.join(path, 'parameters.txt'))
    options = OptionParser().load_options(os.path.join(path, 'config.txt'))
    initial = InitialConditionParser().load_initial_condition(os.path.join(path, 'initial.txt'), parameters=parameters)

    if model.k_lock:
        model.bake_k(parameters=parameters)
    
    initial_condition = model.make_initial_condition(initial)

    return SimulationSpecification(model, parameters, options, initial_condition)

if __name__ == '__main__':
    import sys
    print(load(sys.argv[1]))