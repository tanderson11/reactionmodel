import json
from functools import reduce
from dataclasses import dataclass

import yaml
from yaml import SafeLoader as Loader
import numpy as np
import pandas as pd

from reactionmodel.model import Model
import reactionmodel.syntax

@dataclass
class T():
    t_span: tuple
    t_eval: tuple = None

@dataclass
class ParseResults():
    model: Model = None
    parameters: dict = None
    t: T = None
    initial_condition: dict = None
    simulator_config: dict = None

def parse_parameters(parameters_dict):
    parameters = {}
    for p_name, p in parameters_dict.items():
        if isinstance(p, dict):
            p_dict = p.copy()
            path = p_dict.pop('path')
            header = p_dict.pop('header', None)
            value = np.array(pd.read_csv(path, header=header), dtype=float)
        else:
            try:
                value = float(p)
            except ValueError:
                value = p
        parameters[p_name] = value

    return parameters

def parse_initial_condition(families, ic_dict, syntax=reactionmodel.syntax.Syntax()):
    # we get a list of dictionaries with families expanded
    all_entries = syntax.expand_families(families, ic_dict)
    # combine all those entries into single dictionary
    return reduce(lambda x,y: {**x, **y}, all_entries)

@dataclass
class ConfigParser():
    '''A class for parsing configuration dictionaries/files associated with forward simulators.

    This class is intended to be subclassed in packages that implement forward simulation.'''

    key = 'simulator_config'
    @classmethod
    def from_dict(cls, config_dictionary):
        return config_dictionary

    @classmethod
    def load(cls, path, format='yaml'):
        with open(path, 'r') as f:
            if format == 'yaml':
                data = yaml.load(f, Loader=Loader)
            elif format == 'json':
                data = json.load(f)
            else:
                raise ValueError(f"Expected format keyword to be one of 'json' or 'yaml' found {format}")
        return cls.from_dict(data[cls.key])

def loads(data, syntax=reactionmodel.syntax.Syntax(), ConfigParser=ConfigParser):
    used_keys = []

    kwargs = {}

    families = data.get('families', {})
    if set(['species', 'reactions']).issubset(data.keys()):
        used_keys.extend(['species', 'reactions'])
        kwargs['model'] = Model.parse_model(families, data['species'], data['reactions'], syntax=syntax)

    if 'parameters' in data.keys():
        used_keys.append('parameters')
        kwargs['parameters'] = parse_parameters(data['parameters'])

    if 't' in data.keys():
        used_keys.append('t')
        t_data = data['t']
        kwargs['t'] = T(**t_data)

    if 'initial_condition' in data.keys():
        used_keys.append('initial_condition')
        kwargs['initial_condition'] = parse_initial_condition(families, data['initial_condition'], syntax=syntax)

    if 'simulator_config' in data.keys():
        used_keys.append('simulator_config')
        kwargs['simulator_config'] = ConfigParser.from_dict(data['simulator_config'])

    return ParseResults(**kwargs)

def load(*paths, format='yaml', ConfigParser=ConfigParser):
    """Combines yaml/json data from a variety of paths into one dictionary. Then loads a specification from the data."""
    d = {}
    for p in paths:
        d.update(reactionmodel.parser.load_dictionary(p, format=format))

    return loads(d, ConfigParser=ConfigParser)

def load_dictionary(path, format='yaml'):
    with open(path, 'r') as f:
        if format == 'yaml':
            data = yaml.load(f, Loader=Loader)
        elif format == 'json':
            data = json.load(f)
        else:
            raise ValueError(f"Expected format keyword to be one of 'json' or 'yaml' found {format}")
    return data