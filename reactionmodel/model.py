from enum import Enum
from dataclasses import dataclass
from dataclasses import asdict
from functools import cached_property
from keyword import iskeyword
import re

from typing import NamedTuple
from types import FunctionType as function

import numpy as np
import scipy.sparse
from simpleeval import simple_eval

import reactionmodel.syntax
import reactionmodel.propensity

NO_NUMBA = False
try:
    import numba
    from numba.core.registry import CPUDispatcher
except ModuleNotFoundError:
    NO_NUMBA = True

def find_duplicates(collection):
    s = set()
    duplicates = set()
    for x in collection:
        if x not in s:
            s.add(x)
            continue
        duplicates.add(x)
    return duplicates

string_sub_pattern = re.compile('\~(.+?)\~')
# in the context of {'y': 'foo'}, replace strings like "x * ~y~" with "x * foo"
def process_string_substitutions(expression, parameters):
    try:
        processed = re.sub(string_sub_pattern, lambda m: '(' + str(simple_eval(m.group(1), names=parameters)) + ')', expression)
    except:
        print("oh no")
    return processed

def eval_expression(expression, parameters):
    """Evaluate lazily defined parameter as expression in the context of the provided parameters."""
    if isinstance(expression, (float, np.floating)):
        return expression

    print(f"Evaluating expression: {expression} =>", end=" ")

    if '~' in expression:
        print(f"Substituting evalutions: {expression} =>", end=" ")
        expression = process_string_substitutions(expression, parameters)
        print(f"{expression} =>", end=" ")

    for k in parameters.keys():
        if iskeyword(k):
            raise ValueError(f"parameter named {k} conflicted with Python keyword")

    if not isinstance(parameters, dict):
        parameters = parameters.asdict()
    evaluated = simple_eval(expression, names=parameters)
    try:
        evaluated = float(evaluated)
    except ValueError as exc:
        # try to evaluate AGAIN in case we just found more lazy parameters inside of matrices
        try:
            evaluted_twice = simple_eval(evaluated, names=parameters)
            evaluated = float(evaluted_twice)
        except ValueError:
            raise ValueError(f"Python evaluation of the string {expression} did not produce a float literal (produced {evaluated}). Tried evaluating twice in case of doubly-lazy dictionary and got {evaluted_twice}") from exc
    except Exception as e:
        print(expression, evaluated)
        raise(e)
    print(evaluated)
    return evaluated

def _eval_matrix(matrix_name, parameters):
    matrix = parameters[matrix_name]
    evaluated = np.zeros_like(matrix)
    it = np.nditer(matrix, flags=['refs_ok', 'multi_index'])
    for exp in it:
        evaluated[it.multi_index] = eval_expression(str(exp), parameters)
    return evaluated

@dataclass(frozen=True)
class Species():
    """A species (i.e. object) in the physical system."""
    name: str
    description: str = ''

    def __str__(self):
        return self.name

    def __format__(self, __format_spec: str) -> str:
        return format(str(self), __format_spec)

    def to_dict(self):
        """Return dictionary representation of self."""
        selfdict = asdict(self)
        if not self.description:
            selfdict.pop('description')
        return selfdict

    @classmethod
    def from_dict(cls, d):
        """Create a Species from a dictionary representation."""
        return cls(**d)

class MultiplicityType(Enum):
    """A way that a Species could be involved in a Reaction."""
    reacants = 'reactants'
    products = 'products'
    stoichiometry = 'stoichiometry'
    kinetic_order = 'kinetic order'

@dataclass(frozen=True)
class RatelessReaction():
    """A reaction without rate information. Useful as a standalone in agent-based simulation to trigger an event."""
    reactants: tuple[Species]
    products: tuple[Species]
    description: str = ''
    reversible: bool = False

    def __post_init__(self):
        """Adjust everything that should be a tuple to be a tuple."""
        if isinstance(self.reactants, Species) or isinstance(self.reactants, tuple):
            object.__setattr__(self, 'reactants', (self.reactants,))
        if not isinstance(self.reactants, tuple):
            object.__setattr__(self, 'reactants', tuple(self.reactants))
        if isinstance(self.products, Species) or isinstance(self.products, tuple):
            object.__setattr__(self, 'products', (self.products,))
        if not isinstance(self.products, tuple):
            object.__setattr__(self, 'products', tuple(self.products))
        assert(isinstance(self.reactants, tuple))
        assert(isinstance(self.products, tuple))

        for reactant in self.reactants:
            assert(isinstance(reactant, (tuple, Species)))
        for product in self.products:
            assert(isinstance(product, (tuple, Species)))

        self.verify_integer(self.reactants)
        self.verify_integer(self.products)

        assert len(self.reactants) == len(set(self.reactants)), "Noticed duplicated reactants. Instead of reactants=[X, X] try reactants=[(X, 2)]"
        assert len(self.products) == len(set(self.products)), "Noticed duplicated products. Instead of products=[X, X] try products=[(X, 2)]"

        assert self.reversible is False, "Reversible reactions are not supported. Create separate forward and back reactions instead."

    @staticmethod
    def verify_integer(species_set):
        for t in species_set:
            if not isinstance(t, tuple):
                continue
            multiplicity = t[1]
            assert(isinstance(multiplicity, (int, np.integer)))

    def to_dict(self, keep_species_objects=False):
        """Return dictionary representation of self."""
        selfdict = {}
        if self.description:
            selfdict['description'] = self.description
        if not keep_species_objects:
            selfdict['reactants'] = [r.name if isinstance(r, Species) else (r[0].name, r[1]) for r in self.reactants]
            selfdict['products']  = [p.name if isinstance(p, Species) else (p[0].name, p[1]) for p in self.products ]
        else:
            selfdict['reactants'] = list(self.reactants)
            selfdict['products']  = list(self.products)

        if self.reversible:
            selfdict['reversible'] = self.reversible

        return selfdict

    def order(self):
        order = 0
        for rt in self.kinetic_orders:
            multiplicity = rt[1] if isinstance(rt, tuple) else 1
            order += multiplicity
        return order

    @cached_property
    def reactant_species(self):
        """The set of Species involved as reactants."""
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.reactants])

    @cached_property
    def product_species(self):
        """The set of Species involved as products."""
        return set([(p[0] if isinstance(p, tuple) else p) for p in self.products])

    def _get_positive_negative_multiplicity_data(self, mult_type):
        positive_multplicity_data = []
        negative_multiplicity_data = []
        if mult_type == MultiplicityType.reacants:
            positive_multplicity_data = self.reactants
        elif mult_type == MultiplicityType.products:
            positive_multplicity_data = self.products
        elif mult_type == MultiplicityType.stoichiometry:
            positive_multplicity_data = self.products
            negative_multiplicity_data = self.reactants
        else:
            raise ValueError(f"bad value for type of multiplicities to calculate: {mult_type}.")

        return positive_multplicity_data, negative_multiplicity_data

    def multiplicities(self, mult_type):
        """Return list of multiplicity of each species in reaction for multiplicity type (e.g. stoichiometry)"""
        multplicities = {}

        positive_multplicity_data, negative_multiplicity_data = self._get_positive_negative_multiplicity_data(mult_type)

        for species in negative_multiplicity_data:
            if isinstance(species, tuple):
                species, multiplicity = species
            else:
                multiplicity = 1
            multplicities[species] = -1 * multiplicity

        for species in positive_multplicity_data:
            if isinstance(species, tuple):
                species, multiplicity = species
            else:
                multiplicity = 1
            try:
                multplicities[species] += multiplicity
            except KeyError:
                multplicities[species] = multiplicity

        return multplicities

    @classmethod
    def rebuild_multiplicity(cls, species_context, multiplicity_list):
        """When loading a Reaction from strings, replace a species name with the Species in species_context in a multiplicity list."""
        multiplicity_info = []
        for species_info in multiplicity_list:
            # we want to put the actual species object into the dict
            if isinstance(species_info, (tuple, list)):
                species_info, multiplicity = species_info
                species_name = species_info if isinstance(species_info, str) else species_info['name']
                multiplicity_info.append((species_context[species_name], multiplicity))
                continue
            # in some formats we will see the lists with just an abbreviation/name for species
            species_name = species_info if isinstance(species_info, str) else species_info['name']
            multiplicity_info.append(species_context[species_name])
        return multiplicity_info

    @classmethod
    def _preprocess_dictionary_for_loading(cls, dictionary, species_context):
        reactants = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=dictionary['reactants']
        )
        products = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=dictionary['products']
        )

        reaction = dictionary.copy()
        reaction['products'] = products
        reaction['reactants'] = reactants

        return reaction

    @classmethod
    def from_dict(cls, dictionary, species_context):
        """Given dictionary representation of a Reaction and a species_context of Species objects, return a Reaction"""
        reaction_dict = cls._preprocess_dictionary_for_loading(dictionary, species_context)
        return cls(**reaction_dict)

    def stoichiometry(self):
        """Return the multiplicity of each species in stoichiometry of the reaction."""
        return self.multiplicities(MultiplicityType.stoichiometry)

    def used(self):
        """Return the set of all Species involved in some way in this Reaction."""
        return self.product_species.union(self.reactant_species)

    def __repr__(self) -> str:
        return (f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products})")

    def __str__(self) -> str:
        return f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products})"

@dataclass(frozen=True)
class Reaction(RatelessReaction):
    """A reaction that converts reactants to products."""
    kinetic_orders: tuple[Species] = None
    k: float = None

    def __post_init__(self):
        """Add default kinetic orders if unspecified."""
        super().__post_init__()

        if (not isinstance(self.kinetic_orders, type(None))) and len(self.kinetic_orders) != 0:
            object.__setattr__(self, 'kinetic_orders', tuple(self.kinetic_orders))
        else:
            object.__setattr__(self, 'kinetic_orders', None)
        assert(isinstance(self.kinetic_orders, (tuple, type(None)))), self.kinetic_orders

        # None or zero length
        if self.kinetic_orders is None:
            object.__setattr__(self, 'kinetic_orders', self.reactants)

    def used(self):
        """Return the set of all Species involved in some way in this Reaction."""
        return super().used().union(self.kinetic_order_species)

    def to_dict(self, keep_species_objects=False):
        """Return dictionary representation of self."""
        selfdict = super().to_dict(keep_species_objects=keep_species_objects)

        if self.kinetic_orders != self.reactants:
            selfdict['kinetic_orders'] = self.kinetic_orders

        selfdict['k'] = self.k
        return selfdict

    @classmethod
    def _preprocess_dictionary_for_loading(cls, dictionary, species_context):
        reaction = super()._preprocess_dictionary_for_loading(dictionary, species_context)

        kinetic_orders_info = dictionary.get('kinetic_orders', {})
        kinetic_orders = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=kinetic_orders_info
        )
        reaction['kinetic_orders'] = kinetic_orders
        return reaction

    def _get_positive_negative_multiplicity_data(self, mult_type):
        if mult_type != MultiplicityType.kinetic_order:
            return super()._get_positive_negative_multiplicity_data(mult_type)

        positive_multplicity_data = self.kinetic_orders
        negative_multiplicity_data = []

        return positive_multplicity_data, negative_multiplicity_data

    @cached_property
    def kinetic_order_species(self):
        """The set of species involved in the rate law."""
        if self.kinetic_orders is None:
            return set([])
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.kinetic_orders])

    def kinetic_order(self):
        """Return the kinetic intensity of each species in the reaction."""
        return self.multiplicities(MultiplicityType.kinetic_order)

    def eval_k_with_parameters(self, parameters):
        """Evaluate lazy rate constant in the context of parameters."""
        k = eval_expression(self.k, parameters)
        return k

    def __repr__(self) -> str:
        return (f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products}, kinetic_order={self.kinetic_orders}, k={self.k})")

    def __str__(self) -> str:
        return f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products}, kinetic_order={self.kinetic_orders}, k={self.k})"

class RateConstantCluster(NamedTuple):
    k: function
    slice_bottom: int
    slice_top: int

@dataclass(frozen=True)
class ReactionRateFamily():
    reactions: tuple[Reaction]
    k: callable

    def __post_init__(self):
        if isinstance(self.reactions, list):
            object.__setattr__(self, 'reactions', tuple(self.reactions))
        assert(isinstance(self.reactions, tuple))

    def used(self):
        used = set()
        for r in self.reactions:
            for s in r.used():
                used.add(s)
        return used

    def to_dict(self):
        return {'reactions': [asdict(r) for r in self.reactions], 'k': self.k.__name__}

    @classmethod
    def from_dict(cls, reactions, species_context, k, reaction_class=Reaction):
        self_dict = {}
        our_reactions = []
        for r_dict in reactions:
            our_reactions.append(reaction_class.from_dict(r_dict, species_context))

        self_dict['reactions'] = our_reactions
        self_dict['k'] = k

        return cls(**self_dict)

class UnusedSpeciesError(Exception):
    """Model was provided a Species that was used in 0 Reactions."""

class MissingParametersError(Exception):
    """A lazy rate constant needed to be used but no parameters were provided for evaluation."""

class Model():
    """A model of a physical system consisting of Species and Reactions.

    Attributes
    ----------
    n_species: int
        Number of species in model.
    n_reactions: int
        Number of reactions in model.
    species: list[Species]
        Ordered list of Species in model.
    all_reactions: list[Reaction]
        Ordered list of all Reactions in model.
    reaction_groups: list[Reaction, ReactionRateFamily]
        Ordered list of all reaction groups, where reactions are grouped together if the
        evaluation of their time dependent rate constants will be done in vectorized fashion.
    reject_duplicates: bool, optional
        If True, raise an error if the model has duplicated reactions. Defaults to True.

    Methods
    ----------
    stoichiometry():
        Return stoichiometry matrix N. N_ij = stoichiometry of Species i in Reaction j.
    kinetic_order():
        Return matrix of kinetic orders X. X_ij = intensity of Species i in rate law of Reaction j.
    get_k(reaction_to_k=None, parameters=None, jit=False)
        Return a function k(t) that gives the rate constants for reactions at time t.
    get_dydt(reaction_to_k=None, parameters=None, jit=False)
        Return a function dydt(t, y) that gives the time derivative of species quantities at time t.
    """
    has_hooks=False
    def __init__(self, species: list[Species], reactions: list[Reaction], reject_duplicates=True) -> None:
        """Make a model for the given species and reactions.

        Parameters
        ----------
        species : list[Species]
            An ordered list of species. Defines the order of matrix rows.
        reactions : list[Reaction]
            An ordered list of reactions. Defines the order of matrix columns.

        Raises
        ------
        ValueError
            If no species are specified.
        ValueError
            If no reactions are specified.
        ValueError
            If species are not unique.
        ValueError
            If reactions are not unique
        TypeError
            If each reaction is not either a Reaction or a ReactionRateFamily.
        UnusedSpeciesError
            If a given species is not used in any reaction.
        """
        if isinstance(reactions, (Reaction,ReactionRateFamily)):
            reactions = [reactions]
        if len(reactions) == 0:
            raise ValueError("reactions must include at least one reaction.")
        if isinstance(species, Species):
            species = [species]
        if len(species) == 0:
            raise ValueError("species must include at least one species.")
        if reject_duplicates:
            if len(species) != len(set(species)):
                raise ValueError(f"expected all species to be unique. Duplicates: {find_duplicates(species)}")
            if len(reactions) != len(set(reactions)):
                raise ValueError(f"expected all reactions to be unique Duplicates: {find_duplicates(reactions)}")
        self.species = species
        self.all_reactions = []
        self.reaction_groups = []
        used_species = set()
        for r in reactions:
            for s in r.used():
                used_species.add(s)
            if isinstance(r, Reaction):
                self.reaction_groups.append(r)
                self.all_reactions.append(r)
            elif isinstance(r, ReactionRateFamily):
                self.reaction_groups.append(r)
                self.all_reactions.extend(r.reactions)
            else:
                raise TypeError(f"bad type for reaction in model: {type(r)}. Expected Reaction or ReactionRateFamily")

        self.n_species = len(self.species)
        self.n_reactions = len(self.all_reactions)

        self.species_index = {s:i for i,s in enumerate(self.species)}
        self.species_name_index = {s.name:i for i,s in enumerate(self.species)}
        self.reaction_index = {r:i for i,r in enumerate(self.all_reactions)}

        for s in self.species:
            if s not in used_species:
                raise UnusedSpeciesError(f'species {s} is not used in any reactions')

    def to_dict(self):
        return {'species': [s.to_dict() for s in self.species], 'reactions': [r.to_dict() for r in self.reaction_groups]}

    def y_to_dict(self, y):
        y_dict = {s:y[i] for s,i in self.species_name_index.items()}
        return y_dict

    def save(self, filename, format='yaml'):
        if format=='yaml':
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(self.to_dict(), f, Dumper=yaml.SafeDumper, sort_keys=False)
        elif format=='json':
            import json
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")

    @classmethod
    def from_dict(cls, dictionary, functions_by_name=None, species_class=Species, reaction_class=Reaction, reaction_rate_family_class=ReactionRateFamily):
        if functions_by_name is None:
            functions_by_name = {}
        dictionary = dictionary.copy()
        species = {}
        for species_dict in dictionary.pop('species'):
            s = species_class(**species_dict)
            species[s.name] = s

        reactions = []
        for reaction_dict in dictionary.pop('reactions'):
            # is it a ReactionRateFamily?
            if 'reactions' in reaction_dict.keys():
                try:
                    reactions.append(reaction_rate_family_class.from_dict(reaction_dict['reactions'], species, functions_by_name[reaction_dict['k']], reaction_class=reaction_class))
                except KeyError as exc:
                    raise KeyError(f'failed to find function with name {reaction_dict["k"]} when loading a Model that has a ReactionRateFamily. Did you a pass the keyword argument functions_by_name?') from exc
                continue
            try:
                reactions.append(reaction_class.from_dict(reaction_dict, species))
            except Exception as e:
                print(f"While trying to parse reaction {reaction_dict} ...")
                raise(e)

        species = [s for s in species.values()]
        return cls(species, reactions, **dictionary)


    def __eq__(self, other: object) -> bool:
        if tuple(self.species) != tuple(other.species):
            return False
        if set(self.all_reactions) != set(other.all_reactions):
            return False
        return True

    def get_k(self, reaction_to_k: dict=None, parameters: dict=None, jit=False):
        """Return a function k(t, y=None) => rate constants of reactions at time t and state y. In general, k should have no functional dependency on y.

        Parameters
        ----------
        reaction_to_k : dict, optional
            A mapping of reaction: k float or k(t) for reactions
            where k was not already specified, by default None
        parameters : dict, optional
            A mapping of parameter_name: value for evaluating lazily defined rate constants,
            by default None
        jit : bool, optional
            If True, use numba.jit(nopython=True) to return low-level callable (i.e. faster)
            version of k(t), by default False.

        Returns
        -------
        Callable or np.ndarray
            The function k(t, y) => rate constants of reactions at time t and state y. Returns array of constants
            if no rate constant has explicit time dependence and no state dependence.

        Raises
        ------
        KeyError
            If a Reaction has k=None and is not provided k in reaction_to_k.
        MissingParametersError
            If a Reaction has a lazily defined rate law in terms of parameter that is not
            given in parameters.
        TypeError
            If a Reaction has a rate constant of an invalid type (i.e. not a float or string).
        ModuleNotFoundError
            If jit=True but numba is not installed.
        """
        if reaction_to_k is None:
            reaction_to_k = {}
        initial_reaction_to_k = reaction_to_k.copy()
        for r in self.reaction_groups:
            if r.k is not None:
                assert r not in initial_reaction_to_k.keys(), f"The rate constant for reaction {r} was already defined as {r.k} but it was also supplied as {reaction_to_k[r]} at runtime."
                # now it's safe to use the presupplied value of k as there is no conflict
                reaction_to_k[r] = r.k

        # ReactionRateFamilies allow us to calculate k(t) for a group of reactions all at once
        base_k = np.zeros(self.n_reactions)
        k_families = []
        i = 0
        # reactions not self.reactions so we see families
        for r in self.reaction_groups:
            try:
                k = reaction_to_k[r]
            except KeyError as exc:
                raise KeyError(f"for reaction{r}, k was unset and it was also not supplied at runtime in the `reaction_to_k` dictionary.") from exc
            # in __init__ we guranteed that one of the following is True:
            # isinstance(r, Reaction) or isinstance(r, ReactionRateFamily)
            if isinstance(r, ReactionRateFamily):
                k_families.append(RateConstantCluster(k, i, i+len(r.reactions)+1))
                i += len(r.reactions)
                continue

            # Only reachable if isinstance(r, Reaction)
            assert(isinstance(r,Reaction))
            if isinstance(k, str):
                if parameters is None:
                    raise MissingParametersError("attempted to get k(t) without a parameter dictionary where at least one rate constant was a lazy expression that needs a parameter dictionary to be evaluated")
                base_k[i] = r.eval_k_with_parameters(parameters)
            elif isinstance(k, (int, float)):
                base_k[i] = float(k)
            elif isinstance(k, function) or (isinstance(k, CPUDispatcher)):
                if isinstance(k, CPUDispatcher) and not jit:
                    print("WARNING: one k was a numba compiled function, but jit was not set equal to True when calling Model.get_k(). May not reach maximum performance.")
                k_families.append(RateConstantCluster(k, i, i+1))
            else:
                raise TypeError(f"a reaction's rate constant should be, a float, a string expression (evaluated --> float when given parameters), or function with signature k(t) --> float. Found: {k}")

            i+=1

        if jit:
            if NO_NUMBA:
                raise ModuleNotFoundError("""No module named 'numba'. To use jit=True functions, you must install this package with extras. Try `poetry add "reactionmodel[extras]"` or `pip install "reactionmodel[extras]".""")
            # convert families into relevant lists
            k = self._get_k_jit(np.array(base_k), k_families)
            return k

        return self._get_k(base_k, k_families)

    def _get_k_jit(self, base_k, k_families):
        # k_jit can't be an ordinary method because we won't be able to have `self` as an argument in nopython
        # but needs to close around various properties of self, so we define as a closure using this factory function

        # if we have no explicit time dependence, just return the rate constants
        if len(k_families) == 0:
            return base_k.copy()

        # otherwise, we have to apply the familiy function to differently sized blocks
        k_functions, k_slice_bottoms, k_slice_tops = map(np.array, zip(*k_families))

        if len(k_functions) > 1:
            raise JitNotImplementedError("building a nopython jit function for the rate constants isn't supported with more than 1 subcomponent of k having explicit time dependence. Try using a ReactionRateFamily for all reactions and supplying a vectorized k.")

        # now we have only one subcomponent we have to deal with
        k_function, k_slice_bottom, k_slice_top = k_functions[0], k_slice_bottoms[0], k_slice_tops[0]

        @numba.jit(nopython=True)
        def k_jit(t, y=None):
            k = base_k.copy()
            k[k_slice_bottom:k_slice_top] = k_function(t, y=y)
            return k

        return k_jit


    def multiplicity_matrix(self, mult_type: MultiplicityType, reactions):
        """For a kind of multiplicity, return a matrix A_ij of multiplicity of Species i in Reaction j."""
        matrix = np.zeros((len(self.species), len(reactions)))
        for column, reaction in enumerate(reactions):
            multiplicity_column = np.zeros(len(self.species))
            reaction_info = reaction.multiplicities(mult_type)
            for species, multiplicity in reaction_info.items():
                multiplicity_column[self.species_index[species]] = multiplicity

            matrix[:,column] = multiplicity_column

        return matrix

    def _get_k(self, base_k, k_families):
        if len(k_families) == 0:
            return base_k.copy()
        def k(t, y=None):
            k = base_k.copy()
            for family in k_families:
                kt = family.k(t, y=y)
                if kt.shape != k[family.slice_bottom:family.slice_top].shape: raise ValueError(f"k function of ReactionRateFamily {family} failed to produce output of the right size")
                k[family.slice_bottom:family.slice_top] = kt
            return k
        return k

    def legend(self):
        return [s.name for s in self.species]

    def stoichiometry(self):
        """Return stoichiometry matrix N of model.

        N_ij = stoichiometry of Species i in Reaction j"""
        return self.multiplicity_matrix(MultiplicityType.stoichiometry, self.all_reactions)

    def kinetic_order(self):
        """Return kinetic intensity matrix X of model.

        X_ij = kinetic intensity of Species i in Reaction j"""
        return self.multiplicity_matrix(MultiplicityType.kinetic_order, self.all_reactions)

    def get_propensities_function(self, reaction_to_k: dict=None, parameters: dict=None, jit: bool=False, sparse: bool=False):
        """Return function a(t, y) that returns the propensity of each reaction at time t given state=y.

        Parameters
        ----------
        reaction_to_k : dict, optional
            A mapping of reaction: k float or k(t) for reactions
            where k was not already specified, by default None
        parameters : dict, optional
            A mapping of parameter_name: value for evaluating lazily defined rate constants,
            by default None
        jit : bool, optional
            If True, use numba.jit(nopython=True) to return low-level callable (i.e. faster)
            version of k(t), by default False.

        Returns
        -------
        Callable
            A function a(t, y) that returns the propensity of each reaction at time t given state=y.
            Reaction r's propensity a_r(t, y) is such that in time [t, t+dt) the reaction fires
            once with probability a_r(t, y).
        """
        if jit:
            if sparse: raise ValueError("both jit=True and sparse=True is not supported while getting propensity function")
        return reactionmodel.propensity.construct_propensity_function(self.get_k(parameters=parameters, reaction_to_k=reaction_to_k, jit=jit), self.kinetic_order(), jit=jit)

    def get_dydt(self, reaction_to_k: dict=None, parameters: dict=None, jit: bool=False, sparse=False):
        """Returns dydt(t,y), a function that calculates the time derivative of the system.

        Parameters
        ----------
        reaction_to_k : dict, optional
            A mapping of reaction: k float or k(t) for reactions
            where k was not already specified, by default None
        parameters : dict, optional
            A mapping of parameter_name: value for evaluating lazily defined rate constants,
            by default None
        jit : bool, optional
            If True, use numba.jit(nopython=True) to return low-level callable (i.e. faster)
            version of k(t), by default False.

        Returns
        -------
        dydt
            The function dydt(t, y) that gives the time derivative of a system in state y at time t.
        """
        if jit:
            if sparse: raise ValueError("both jit=True and sparse=True is not possible whiile getting dydt")
            return self._get_jit_dydt_function(reaction_to_k=reaction_to_k, parameters=parameters)
        return self._get_dydt(reaction_to_k=reaction_to_k, parameters=parameters, sparse=sparse)

    def _get_dydt(self, reaction_to_k=None, parameters=None, sparse=False):
        calculate_propensities = self.get_propensities_function(reaction_to_k=reaction_to_k, parameters=parameters, sparse=sparse)
        N = self.stoichiometry()
        if sparse:
            N = scipy.sparse.csr_array(N)
        def dydt(t, y):
            propensities = calculate_propensities(t, y)

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt = N @ propensities
            return dydt
        return dydt

    def _get_jit_dydt_function(self, reaction_to_k=None, parameters=None):
        jit_calculate_propensities = self._get_jit_propensities_function(reaction_to_k=reaction_to_k, parameters=parameters)
        N = self.stoichiometry()
        @numba.jit(nopython=True)
        def jit_dydt(t, y, propensity_function):
            propensities = jit_calculate_propensities(t, y)

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt = N @ propensities
            return dydt

        return jit_dydt

    def make_initial_condition(self, dictionary: dict, parameters: dict=None):
        """Given a dictionary of species_name: quantity, return the state of the system.

        Parameters
        ----------
        dictionary : dict
            A mapping of species_name: quantity (float or string, see parameters arg)
        parameters : dict, optional
            A mapping of parameter_name: value used to evaluated all string
            expressions in dictionary, by default None

        Returns
        -------
        np.ndarray
            A vector of quantities of each species in the system.
        """
        if parameters is None:
            parameters = {}
        x0 = np.zeros(self.n_species)
        for k,v in dictionary.items():
            if isinstance(v, str):
                v = eval_expression(v, parameters)
            x0[self.species_name_index[k]] = float(v)
        return x0

    def get_simulator(self, simulator_class, reaction_to_k=None, parameters=None, jit=False, **simulator_kwargs):
        return simulator_class(
            self.get_k(reaction_to_k=reaction_to_k, parameters=parameters, jit=jit),
            self.stoichiometry(),
            self.kinetic_order(),
            **simulator_kwargs
        )

    @classmethod
    def parse_model(cls, families, species, reactions, syntax=reactionmodel.syntax.Syntax(), **kwargs):
        all_species = []
        for s in species:
            all_species.extend(syntax.expand_families(families, s))
        all_reactions = []
        for r in reactions:
            all_reactions.extend(syntax.expand_families(families, r))
        model_dict = {
            'species'  : all_species,
            'reactions': all_reactions,
        }
        return cls.from_dict(model_dict, **kwargs)

    @classmethod
    def load(cls, filename, format='yaml', syntax=reactionmodel.syntax.Syntax()):
        if format=='yaml':
            import yaml
            with open(filename, 'r') as f:
                d = yaml.load(f, Loader=yaml.SafeLoader)
        elif format=='json':
            import json
            with open(filename, 'r') as f:
                d = json.load(f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")
        families = d.get('families', {})
        assert isinstance(families, dict), "families should be a dictionary. In YAML, be careful not to include '-' on lines introducing families."
        return cls.parse_model(families, d['species'], d['reactions'], syntax=syntax)

    def __repr__(self) -> str:
        return f'Model with {len(self.species)} species and {len(self.all_reactions)} reactions.'

class JitNotImplementedError(Exception):
    """Could not craft numba.jit function because of limitations of numba's nopython mode."""
