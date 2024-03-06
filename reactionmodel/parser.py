import yaml
from yaml import SafeLoader as Loader
import json
import numpy as np
import pandas as pd
from functools import reduce

from numpy.typing import ArrayLike
from itertools import product
from enum import Enum
from dataclasses import dataclass

from reactionmodel.model import Model


@dataclass(frozen=True)
class Syntax():
    family_denoter = '$'
    family_enumerator = '#'

def enumerated_product(members):
    yield from zip(product(*(range(len(x)) for x in members)), product(*members))

def family_replace(family_names, idx, chosen_members, value, syntax=Syntax(), do_nestings=False):
    for family_name, member, i in zip(family_names, chosen_members, idx):
        if do_nestings and isinstance(value, (tuple, list)):
            value = [family_replace(family_names, idx, chosen_members, v, syntax, do_nestings) for v in value]
            continue
        if do_nestings and isinstance(value, dict):
            value = {k:family_replace(family_names, idx, chosen_members, v, syntax, do_nestings) for k,v in value.items()}
            continue
        if isinstance(value, (int, float)):
            continue
        value = value.replace(syntax.family_denoter + family_name, member)
        value = value.replace(syntax.family_enumerator + family_name, str(i))
    return value

class ReactionRateFamilyApplicationMethod(Enum):
    group = 'group'
    split = 'split'

def expand_families(families, atom, syntax=Syntax()):
    try:
        used_families = atom.pop('used_families')
    except KeyError:
        return [atom]
    nested_fields = ['products', 'reactants']
    # are we a ReactionRateFamily?
    if 'reactions' in atom.keys():
        # we are a ReactionRateFamily
        try:
            family_method = ReactionRateFamilyApplicationMethod(atom.pop('family_method'))
        except KeyError:
            family_method = ReactionRateFamilyApplicationMethod.group
        
        if family_method == ReactionRateFamilyApplicationMethod.group:
            for field, value in atom.items():
                if field != 'reactions':
                    assert syntax.family_denoter not in value
                    assert syntax.family_enumerator not in value
                    continue

            new_reactions = []
            for r in atom['reactions']:
                r_ = r.copy()
                r_['used_families'] = used_families
                new = expand_families(families, r_, syntax)
                print(new)
                new_reactions.extend(new)
            new_rrf = atom.copy()
            new_rrf['reactions'] = new_reactions
            return new_rrf
        elif family_method == ReactionRateFamilyApplicationMethod.split:
            # we want to continue with expansion, but make a note that reactions will have to be expanded as well
            nested_fields.append('reactions')
    family_members = []
    for _, global_family_name in used_families.items():
        members = families[global_family_name]
        # append the *list*, so we can keep our families straight
        family_members.append(members)

    new_atoms = []
    for idx, combination in enumerated_product(family_members):
        new_atom = {}
        for field, value in atom.items():
            new_field = family_replace(used_families, idx, combination, field, syntax=syntax, do_nestings=False)
            new_atom[new_field] = family_replace(used_families, idx, combination, value, syntax=syntax, do_nestings=(field in nested_fields))
        new_atoms.append(new_atom)
    
    return new_atoms

def parse_model(families, species, reactions, syntax=Syntax()):
    all_species = []
    for s in species:
        all_species.extend(expand_families(families, s, syntax=syntax))
    all_reactions = []
    for r in reactions:
        all_reactions.extend(expand_families(families, r, syntax=syntax))
    model_dict = {
        'species'  : all_species,
        'reactions': all_reactions,
    }
    print(model_dict)
    return Model.from_dict(model_dict)

@dataclass
class ParseResults():
    model: Model 
    parameters: dict
    initial_condition: dict
    simulator_config: dict

@dataclass
class Parameter():
    name: str
    value: float

    @classmethod
    def from_dict(cls, dictionary):
        dictionary = dictionary.copy()

        path = dictionary.pop('path', None)
        if path is not None:
            header = dictionary.pop('header', None)
            dictionary['value'] = np.array(pd.read_csv(path, header=header))

        try:
            dictionary['value'] = float(dictionary['value'])
        except TypeError:
            pass

        return cls(**dictionary)

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

def parse_initial_condition(families, ic_dict, syntax=Syntax()):
    # we get a list of dictionaries with families expanded
    all_entries = expand_families(families, ic_dict, syntax=syntax)
    # combine all those entries into single dictionary
    return reduce(lambda x,y: {**x, **y}, all_entries)

def loads(data, syntax=Syntax()):
    model = None
    parameters = None
    initial_condition = None
    simulator_config = None

    families = data.get('families', {})
    if set(['species', 'reactions']).issubset(data.keys()):
        model = parse_model(families, data['species'], data['reactions'], syntax=syntax)

    if 'parameters' in data.keys():
        parameters = parse_parameters(data['parameters'])

    if 'initial_condition' in data.keys():
        initial_condition = parse_initial_condition(families, data['initial_condition'], syntax=syntax)

    if 'simulator_config' in data.keys():
        simulator_config = data['simulator_config']

    return ParseResults(model, parameters, initial_condition, simulator_config)

def load(path, format='yaml'):
    with open(path, 'r') as f:
        if format == 'yaml':
            data = yaml.load(f, Loader=Loader)
        elif format == 'json':
            data = json.load(f)
        else:
            raise ValueError(f"Expected format keyword to be one of 'json' or 'yaml' found {format}")
    return loads(data)