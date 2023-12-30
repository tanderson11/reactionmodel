from dataclasses import dataclass

from reactionmodel.parser import Parser, AtomFactory, Property, RichProperty, FamilyFactory
from reactionmodel.msl import ParameterFactory, DerivedParameterFactory, PathProperty
from reactionmodel.util import FrozenDictionary

@dataclass(frozen=True, eq=False)
class Configuration(FrozenDictionary):
    subclass_name = 'SpecifiedOptions'

@dataclass(frozen=True, eq=False)
class InitialCondition(FrozenDictionary):
    subclass_name = 'SpecifiedInitialCondition'

    @staticmethod
    def handle_field(k, v):
        return (float, float(v))

class Option():
    def __init__(self, name, value, description='') -> None:
        self.name = name
        self.value = value
        self.description = description

    def __repr__(self) -> str:
        return f'Option(name={self.name}, value={self.value}, description={self.description})'

class PathOptionFactory(Option):
    klass = Option
    header = "PathOption"
    properties = [PathProperty('path'), RichProperty('description', optional=True)]

class OptionFactory(AtomFactory):
    klass = Option
    header = "Option"
    properties = [Property('value'), RichProperty('description', optional=True)]

class OptionParser(Parser):
    option_factories = [OptionFactory, PathOptionFactory]

    def load_options(self, file):
        parameters = self.parse_file(file, self.option_factories)
        straightforward_dictionary = {}
        for name, p in parameters.items():
            straightforward_dictionary[name] = p.value
        return straightforward_dictionary

class InitialConditionFactory(ParameterFactory):
    header = 'Initial'

class DerviedInitialConditionFactory(DerivedParameterFactory):
    header = 'DerivedInitial'

class InitialConditionParser(Parser):
    frozen_klass = Configuration
    initial_condition_factories = [FamilyFactory, InitialConditionFactory, DerviedInitialConditionFactory]

    def load_initial_condition(self, file, parameters=None):
        # if parameters are supplied, evaluate every pending expression
        # otherwise store those *actual match objects* in place in the dictionary, for evaluation later via evaluate_initial_condition
        ics = self.parse_file(file, self.initial_condition_factories)
        straightforward_dictionary = {}
        for name, ic in ics.items():
            if isinstance(ic, DerviedInitialConditionFactory.klass):
                if parameters is None:
                    straightforward_dictionary[name] = ic
                else:
                    ic.evaluate(parameters)
                    straightforward_dictionary[name] = ic.value
            elif isinstance(ic, InitialConditionFactory.klass):
                straightforward_dictionary[name] = ic.value
        return straightforward_dictionary

    @staticmethod
    def evaluate_initial_condition(initial_condition_dictionary, parameters):
        new = {}
        for name, ic in initial_condition_dictionary.items():
            if isinstance(ic, DerviedInitialConditionFactory.klass):
                ic.evaluate(parameters)
                new[name] = ic.value
            else:
                new[name] = ic

        return new