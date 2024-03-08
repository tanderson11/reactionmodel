from enum import Enum
from dataclasses import dataclass
from dataclasses import asdict
from functools import cached_property

from typing import NamedTuple
from types import FunctionType as function

import numpy as np
from scipy.special import binom
from simpleeval import simple_eval

NO_NUMBA = False
try:
    import numba
    from numba.core.registry import CPUDispatcher
except ModuleNotFoundError:
    NO_NUMBA = True

def eval_expression(expression, parameters):
    """Evaluate lazily defined parameter as expression in the context of the provided parameters."""
    print(f"Evaluating expression: {expression} =>", end=" ")

    if not isinstance(parameters, dict):
        parameters = parameters.asdict()
    evaluated = simple_eval(expression, names=parameters)
    try:
        evaluated = float(evaluated)
    except ValueError as exc:
        raise ValueError(f"Python evaluation the string {expression} did not produce a float literal (produced {evaluated})") from exc
    print(evaluated)
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
class Reaction():
    """A reaction that converts reactants to products."""
    reactants: tuple[Species]
    products: tuple[Species]
    description: str = ''
    kinetic_orders: tuple[Species] = None
    reversible: bool = False
    k: float = None

    def __post_init__(self):
        """Ensure everything that should be a tuple is. Add default kinetic orders if unspecified."""
        if isinstance(self.reactants, Species) or isinstance(self.reactants, tuple):
            object.__setattr__(self, 'reactants', (self.reactants,))
        if not isinstance(self.reactants, tuple):
            object.__setattr__(self, 'reactants', tuple(self.reactants))
        if isinstance(self.products, Species) or isinstance(self.products, tuple):
            object.__setattr__(self, 'products', (self.products,))
        if not isinstance(self.products, tuple):
            object.__setattr__(self, 'products', tuple(self.products))
        if not isinstance(self.kinetic_orders, (tuple, type(None))):
            object.__setattr__(self, 'kinetic_orders', tuple(self.kinetic_orders))
        assert(isinstance(self.reactants, tuple))
        assert(isinstance(self.kinetic_orders, (tuple, type(None))))
        assert(isinstance(self.products, tuple))

        for reactant in self.reactants:
            assert(isinstance(reactant, (tuple, Species)))
        for product in self.products:
            assert(isinstance(product, (tuple, Species)))

        # None or zero length
        if not self.kinetic_orders:
            object.__setattr__(self, 'kinetic_orders', self.reactants)
        assert self.reversible is False, "Reversible reactions are not supported. Create separate forward and back reactions instead."

    def to_dict(self):
        """Return dictionary representation of self."""
        selfdict = {}
        if self.description:
            selfdict['description'] = self.description
        selfdict['reactants'] = [r.name for r in self.reactants]
        selfdict['products'] = [p.name for p in self.products]
        if self.kinetic_orders != self.reactants:
            selfdict['kinetic_orders'] = self.kinetic_orders
        if self.reversible:
            selfdict['reversible'] = self.reversible
        selfdict['k'] = self.k
        return selfdict

    @cached_property
    def reactant_species(self):
        """The set of Species involved as reactants."""
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.reactants])

    @cached_property
    def product_species(self):
        """The set of Species involved as products."""
        return set([(p[0] if isinstance(p, tuple) else p) for p in self.products])

    @cached_property
    def kinetic_order_species(self):
        """The set of species involved in the rate law."""
        if self.kinetic_orders is None:
            return set([])
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.kinetic_orders])

    def eval_k_with_parameters(self, parameters):
        """Evaluate lazy rate constant in the context of parameters."""
        k = eval_expression(self.k, parameters)
        return k

    def multiplicities(self, mult_type):
        """Return list of multiplicity of each species in reaction for multiplicity type (e.g. stoichiometry)"""
        multplicities = {}

        positive_multplicity_data = []
        negative_multiplicity_data = []
        if mult_type == MultiplicityType.reacants:
            positive_multplicity_data = self.reactants
        elif mult_type == MultiplicityType.products:
            positive_multplicity_data = self.products
        elif mult_type == MultiplicityType.stoichiometry:
            positive_multplicity_data = self.products
            negative_multiplicity_data = self.reactants
        elif mult_type == MultiplicityType.kinetic_order:
            positive_multplicity_data = self.kinetic_orders
        else:
            raise ValueError(f"bad value for type of multiplicities to calculate: {mult_type}.")

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
    def from_dict(cls, dictionary, species_context):
        """Given dictionary representation of a Reaction and a species_context of Species objects, return a Reaction"""
        reactants = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=dictionary['reactants']
        )
        products = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=dictionary['products']
        )
        kinetic_orders_info = dictionary.get('kinetic_orders', {})
        kinetic_orders = cls.rebuild_multiplicity(
            species_context=species_context,
            multiplicity_list=kinetic_orders_info
        )

        reaction = dictionary.copy()
        reaction['products'] = products
        reaction['reactants'] = reactants
        reaction['kinetic_orders'] = kinetic_orders
        return cls(**reaction)

    def stoichiometry(self):
        """Return the multiplicity of each species in stoichiometry of the reaction."""
        return self.multiplicities(MultiplicityType.stoichiometry)

    def kinetic_order(self):
        """Return the kinetic intensity of each species in the reaction."""
        return self.multiplicities(MultiplicityType.kinetic_order)

    def used(self):
        """Return the set of all Species involved in some way in this Reaction."""
        return self.product_species.union(self.reactant_species).union(self.kinetic_order_species)

    def __repr__(self) -> str:
        return (f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products}, kinetic_order={self.kinetic_orders}, k={self.k})")

    def __str__(self) -> str:
        return f"Reaction(description={self.description}, reactants={self.reactants}, products={self.products}, kinetic_order={self.kinetic_orders}, k={self.k})"

class RateConstantCluster(NamedTuple):
    k: function
    slice_bottom: int
    slice_top: int

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
    def __init__(self, species: list[Species], reactions: list[Reaction]) -> None:
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
        if len(species) != len(set(species)):
            raise ValueError("expected all species to be unique")
        if len(reactions) != len(set(reactions)):
            raise ValueError("expected all reactions to be unique")
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

    def save(self, filename, format='yaml'):
        if format=='yaml':
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(self.to_dict(), f, Dumper=yaml.SafeDumper)
        elif format=='json':
            import json
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")

    @classmethod
    def from_dict(cls, dictionary, functions_by_name=None):
        if functions_by_name is None:
            functions_by_name = {}
        dictionary = dictionary.copy()
        species = {}
        for species_dict in dictionary.pop('species'):
            s = Species(**species_dict)
            species[s.name] = s

        reactions = []
        for reaction_dict in dictionary.pop('reactions'):
            # is it a ReactionRateFamily?
            if 'reactions' in reaction_dict.keys():
                try:
                    reactions.append(ReactionRateFamily.from_dict(reaction_dict['reactions'], species, functions_by_name[reaction_dict['k']]))
                except KeyError as exc:
                    raise KeyError(f'failed to find function with name {reaction_dict["k"]} when loading a Model that has a ReactionRateFamily. Did you a pass the keyword argument functions_by_name?') from exc
                continue
            reactions.append(Reaction.from_dict(reaction_dict, species))

        species = [s for s in species.values()]
        return cls(species, reactions, **dictionary)


    def __eq__(self, other: object) -> bool:
        if tuple(self.species) != tuple(other.species):
            return False
        if set(self.all_reactions) != set(other.all_reactions):
            return False
        return True

    def get_k(self, reaction_to_k: dict=None, parameters: dict=None, jit=False):
        """Return a function k(t) => rate constants of reactions at time t.

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
            The function k(t) => rate constants of reactions at time t. Returns array of constants
            if no rate constant has explicit time dependence.

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
        for r in self.reaction_groups:
            if r.k is not None:
                assert r not in reaction_to_k.keys(), f"The rate constant for reaction {r} was already defined as {r.k} but it was also supplied as {reaction_to_k[r]} at runtime."
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
            elif isinstance(k, function) or (jit and isinstance(k, CPUDispatcher)):
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
        def k_jit(t):
            k = base_k.copy()
            k[k_slice_bottom:k_slice_top] = k_function(t)
            return k

        return k_jit

    def multiplicity_matrix(self, mult_type: MultiplicityType):
        """For a kind of multiplicity, return a matrix A_ij of multiplicity of Species i in Reaction j."""
        matrix = np.zeros((self.n_species, self.n_reactions))
        for column, reaction in enumerate(self.all_reactions):
            multiplicity_column = np.zeros(self.n_species)
            reaction_info = reaction.multiplicities(mult_type)
            for species, multiplicity in reaction_info.items():
                multiplicity_column[self.species_index[species]] = multiplicity

            matrix[:,column] = multiplicity_column

        return matrix

    def _get_k(self, base_k, k_families):
        if len(k_families) == 0:
            return base_k.copy()
        def k(t):
            k = base_k.copy()
            for family in k_families:
                k[family.slice_bottom:family.slice_top] = family.k(t)
            return k
        return k

    def stoichiometry(self):
        """Return stoichiometry matrix N of model.

        N_ij = stoichiometry of Species i in Reaction j"""
        return self.multiplicity_matrix(MultiplicityType.stoichiometry)

    def kinetic_order(self):
        """Return kinetic intensity matrix X of model.

        X_ij = kinetic intensity of Species i in Reaction j"""
        return self.multiplicity_matrix(MultiplicityType.kinetic_order)

    def get_propensities_function(self, reaction_to_k: dict=None, parameters: dict=None, jit: bool=False):
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
            return self._get_jit_propensities_function(reaction_to_k, parameters)
        return self._get_propensities_function(reaction_to_k, parameters)

    def _get_propensities_function(self, reaction_to_k=None, parameters=None):
        k = self.get_k(parameters=parameters, reaction_to_k=reaction_to_k, jit=False)
        def calculate_propensities(t, y):
            if isinstance(k, np.ndarray):
                k_of_t = k
            else:
                k_of_t = k(t)
            # product along column in kinetic order matrix
            # with states raised to power of involvement
            # multiplied by rate constants == propensity
            # dimension of y is expanded to make it a column vector
            return np.prod(binom(np.expand_dims(y, axis=1), self.kinetic_order()), axis=0) * k_of_t
        return calculate_propensities

    def _get_jit_propensities_function(self, reaction_to_k=None, parameters=None):
        kinetic_order_matrix = self.kinetic_order()
        k_jit = self.get_k(reaction_to_k=reaction_to_k, parameters=parameters, jit=True)
        @numba.jit(nopython=True)
        def jit_calculate_propensities(t, y):
            # Remember, we want total number of distinct combinations * k === rate.
            # we want to calculate (y_i kinetic_order_ij) (binomial coefficient)
            # for each species i and each reaction j
            # sadly, inside a numba C function, we can't avail ourselves of scipy's binom,
            # so we write this little calculator ourselves
            intensity_power = np.zeros_like(kinetic_order_matrix)
            for i in range(0, kinetic_order_matrix.shape[0]):
                for j in range(0, kinetic_order_matrix.shape[1]):
                    if y[i] < kinetic_order_matrix[i][j]:
                        intensity_power[i][j] = 0.0
                    elif y[i] == kinetic_order_matrix[i][j]:
                        intensity_power[i][j] = 1.0
                    else:
                        intensity = 1.0
                        for x in range(0, kinetic_order_matrix[i][j]):
                            intensity *= (y[i] - x) / (x+1)
                        intensity_power[i][j] = intensity

            # then we take the product down the columns (so product over each reaction)
            # and multiply that output by the vector of rate constants
            # to get the propensity of each reaction at time t
            k = k_jit(t)
            product_down_columns = np.ones(len(k))
            for i in range(0, len(y)):
                product_down_columns = product_down_columns * intensity_power[i]
            return product_down_columns * k
        return jit_calculate_propensities

    def get_dydt(self, reaction_to_k: dict=None, parameters: dict=None, jit: bool=False):
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
            return self._get_jit_dydt_function(reaction_to_k=reaction_to_k, parameters=parameters)
        return self._get_dydt(reaction_to_k=reaction_to_k, parameters=parameters)

    def _get_dydt(self, reaction_to_k=None, parameters=None):
        calculate_propensities = self.get_propensities_function(reaction_to_k=reaction_to_k, parameters=parameters)
        N = self.stoichiometry()
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
        def jit_dydt(t, y):
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

class JitNotImplementedError(Exception):
    """Could not craft numba.jit function because of limitations of numba."""

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
    def from_dict(cls, reactions, species_context, k):
        self_dict = {}
        our_reactions = []
        for r_dict in reactions:
            our_reactions.append(Reaction.from_dict(r_dict, species_context))

        self_dict['reactions'] = our_reactions
        self_dict['k'] = k

        return cls(**self_dict)
