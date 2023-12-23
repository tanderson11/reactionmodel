import re
import numpy as np
import pandas as pd
from simpleeval import simple_eval
from reactionmodel.model import ReactionRateFamily, Species, Reaction, Model
from reactionmodel.parser import Parser, AtomFactory, ListMatch, ListProperty, Property, RichProperty, ExpressionProperty, PathProperty, FamilyFactory

# model specification language

# Todo:
## Reaction family rates
## Clarify what a Match object does vs what a Property object does
## Use dataclasses for the output of ParameterParser and OptionParser?

## Constraints:
### overlapping family names like x and xx might go wild
### family names must be only alpha

## Wishlist:
### Wait for full evaluation, so some things can be provided later by the user?
### Do some processing as macro expansion?

class SpeciesMultiplicityListMatch(ListMatch):
    def localize_value_with_family_members(self, syntax, families, chosen_members, idx):
        new_value = []
        for v in self.value:
            if isinstance(v, tuple):
                new_value.append((self.localize_string_with_family_members(v[0], syntax, families, chosen_members, idx), v[1]))
            else:
                new_value.append(self.localize_string_with_family_members(v, syntax, families, chosen_members, idx))
        return new_value

    def evaluate_with_existing_atoms(self, existing_atoms):
        actual_list = []

        for v in self.value:
            # tuple like (species_name, multiplicity)
            if isinstance(v, tuple):
                actual_list.append((existing_atoms[v[0]], v[1]))
            else:
                actual_list.append(existing_atoms[v])
        return actual_list

class SpeciesMultiplicityListProperty(ListProperty):
    match_klass = SpeciesMultiplicityListMatch
    species_pattern = re.compile('^([0-9]*)(.*?)$')
    value_pattern_string = '([a-zA-Z0-9{0}{1}]+)$'

    def inject_syntax(self, syntax):
        return self.value_pattern_string.format(re.escape(syntax.list_delimiter), re.escape(syntax.family_denoter))

    def parse(self, line, syntax):
        # gets split to a list by superclass
        # now we format the list as tuples of (species, multplicity) or just species if multiplicity = 1
        property_match = super().parse(line, syntax)
        if property_match is None:
            return property_match

        species_multiplicity_list = []
        for species_string in property_match.value:
            if species_string == syntax.null_set:
                assert(len(property_match.value) == 1, "Null set signifier found in a list with more than 1 element.")
                species_multiplicity_list = []
                break
            species_match = re.match(self.species_pattern, species_string)
            if species_match is None:
                raise BadSpeciesInReactionError(f"couldn't understand {species_string} as a [multiplicity]SpeciesName.")
            multiplicity = species_match[1]
            species = species_match[2]
            if multiplicity:
                species_multiplicity_list.append((species, int(multiplicity)))
            else:
                species_multiplicity_list.append(species)

        property_match.value = species_multiplicity_list
        return property_match

class SpeciesFactory(AtomFactory):
    klass = Species
    header = "Species"
    properties = [RichProperty('description', optional=True)]

class ReactionFactory(AtomFactory):
    # name, description, reactants, products, rate_involvement=None, k=None, reversible=False
    klass = Reaction
    header = "Reaction"
    properties = [SpeciesMultiplicityListProperty('reactants'), SpeciesMultiplicityListProperty('products'), SpeciesMultiplicityListProperty('rate_involvement', optional=True), RichProperty('description', optional=True), ExpressionProperty('k', optional=True)]

class Parameter():
    def __init__(self, name, value, description='') -> None:
        self.name = name
        self.value = np.float64(value)
        self.description = description

    def __repr__(self) -> str:
        return f'Parameter(name={self.name}, value={self.value}, description={self.description})'

class DerivedParameter(Parameter):
    def __init__(self, name, value, description='') -> None:
        self.name = name
        self.value = value
        self.description = description
    
    def evaluate(self, other_parameters):
        value = simple_eval(self.value, names=other_parameters)
        try:
            value = np.float64(value)
        except ValueError:
            raise ValueError(f"Evaluation of Reaction k defined by the string {self.value} did not produce a float literal (produced {value})")
        print(f"Evaluating expression: {self.value} => {value}")

        self.value = value

    def __repr__(self) -> str:
        return f'DerivedParameter(name={self.name}, value={self.value}, description={self.description})'

class ParameterFactory(AtomFactory):
    klass = Parameter
    header = "Parameter"
    properties = [Property('value'), RichProperty('description', optional=True)]

class DerivedParameterFactory(ParameterFactory):
    klass = DerivedParameter
    header = "DerivedParameter"
    properties = [ExpressionProperty('value'), RichProperty('description', optional=True)]

class Matrix():
    def __init__(self, name, matrix) -> None:
        self.name = name
        self.matrix = matrix

    def __repr__(self) -> str:
        return f'Matrix(name={self.name}, matrix={self.matrix})'

class MatrixFactory(AtomFactory):
    klass = Matrix
    header = "Matrix"
    properties = [PathProperty('path')]

    @classmethod
    def from_properties(cls, name, path):
        matrix = np.array(pd.read_csv(path, header=None))
        return cls.klass(name, matrix)

class BadSpeciesInReactionError(Exception):
    pass

class ModelParser(Parser):
    model_factories = [SpeciesFactory, FamilyFactory, ReactionFactory]

    def load_model(self, file, **kwargs):
        atoms = self.parse_file(file, self.model_factories)
        species = [s for s in atoms.values() if isinstance(s, Species)]
        reactions = [r for r in atoms.values() if (isinstance(r, Reaction) or isinstance(r, ReactionRateFamily))]

        model = Model(species, reactions, **kwargs)

        return model

class ParameterParser(Parser):
    parameter_factories = [ParameterFactory, MatrixFactory, DerivedParameterFactory]

    def load_parameters(self, file):
        parameters = self.parse_file(file, self.parameter_factories)
        straightforward_dictionary = {}
        for name, p in parameters.items():
            if isinstance(p, Parameter):
                if isinstance(p, DerivedParameter):
                    p.evaluate(straightforward_dictionary)
                straightforward_dictionary[name] = p.value
            elif isinstance(p, Matrix):
                straightforward_dictionary[name] = p.matrix
        return straightforward_dictionary

if __name__ == '__main__':
    import sys

    for path in sys.argv[1:]:
        if '.model' in path:
            p = ModelParser()
            m = p.load_model(path)
            print(m)
        
        if '.parameters' in path:
            p = ParameterParser()
            parameters = p.load_parameters(path)
            print(parameters)

    m.bake_k(parameters)