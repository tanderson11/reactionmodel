import os

from reactionmodel.option import OptionParser, InitialConditionParser
from reactionmodel.msl import ModelParser, ParameterParser
from reactionmodel.specification import SimulationSpecification

def load_specification(model_path, params_path, config_path, ic_path):
    model = ModelParser().load_model(model_path)
    parameters = ParameterParser().load_parameters(params_path)
    options = OptionParser().load_options(config_path)
    initial = InitialConditionParser().load_initial_condition(ic_path, parameters=parameters)

    return SimulationSpecification(model, parameters, initial, options)

def load(path):
    mpath = os.path.join(path, 'model.txt')
    ppath = os.path.join(path, 'parameters.txt')
    cpath = os.path.join(path, 'config.txt')
    ipath = os.path.join(path, 'initial.txt')

    return load_specification(mpath, ppath, cpath, ipath)

if __name__ == '__main__':
    import sys
    spec = load(sys.argv[1])
    print(spec)
    print(spec.get_frozen())