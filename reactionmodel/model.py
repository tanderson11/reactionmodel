import numpy as np
from scipy.special import binom

from enum import Enum
from typing import NamedTuple
from types import FunctionType as function
from simpleeval import simple_eval
from dataclasses import dataclass
from dataclasses import asdict
from functools import cached_property

NO_NUMBA = False
try:
    from numba import jit, float64
    from numba.types import Array
    from numba.core.registry import CPUDispatcher
except ModuleNotFoundError:
    NO_NUMBA = True

@dataclass(frozen=True)
class Species():
    name: str
    description: str = ''

    def __str__(self):
        return self.name

    def __format__(self, __format_spec: str) -> str:
        return format(str(self), __format_spec)

class MultiplicityType(Enum):
    reacants = 'reactants'
    products = 'products'
    stoichiometry = 'stoichiometry'
    rate_involvement = 'rate involvement'

@dataclass(frozen=True)
class Reaction():
    name: str
    reactants: tuple[Species]
    products: tuple[Species]
    description: str = ''
    rate_involved: tuple[Species] = None
    reversible: bool = False
    k: float = None

    def __post_init__(self):
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

        if self.rate_involved is None:
            object.__setattr__(self, 'rate_involved', self.reactants)
        assert self.reversible == False, "Reversible reactions are not supported. Create separate forward and back reactions instead."

    @cached_property
    def reactant_species(self):
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.reactants])

    @cached_property
    def product_species(self):
        return set([(p[0] if isinstance(p, tuple) else p) for p in self.products])

    @cached_property
    def rate_involved_species(self):
        if self.rate_involved is None:
            return set([])
        return set([(r[0] if isinstance(r, tuple) else r) for r in self.rate_involved])

    def eval_k_with_parameters(self, parameters):
        if not isinstance(parameters, dict):
            parameters = parameters.asdict()
        k = simple_eval(self.k, names=parameters)
        try:
            k = float(k)
        except ValueError:
            raise ValueError(f"Evaluation of Reaction k defined by the string {self.k} did not produce a float literal (produced {k})")
        print(f"Evaluating expression: {self.k} => {k}")
        return k

    def multiplicities(self, mult_type):
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
        elif mult_type == MultiplicityType.rate_involvement:
            positive_multplicity_data = self.rate_involved
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
    def rebuild_multiplicity(cls, existing_species, multiplicity_list):
        multiplicity_info = []
        for species_info in multiplicity_list:
            # we want to put the actual species object into the dict
            if isinstance(species_info, tuple):
                species_info, multiplicity = species_info
                multiplicity_info.append((existing_species[species_info['name']], multiplicity))
                continue
            multiplicity_info.append(existing_species[species_info['name']])
        return multiplicity_info

    @classmethod
    def from_species_and_dictionary(cls, species, dictionary):
        reactants = cls.rebuild_multiplicity(existing_species=species, multiplicity_list=dictionary['reactants'])
        products = cls.rebuild_multiplicity(existing_species=species, multiplicity_list=dictionary['products'])
        rate_involved = cls.rebuild_multiplicity(existing_species=species, multiplicity_list=dictionary['rate_involved'])

        reaction = dictionary.copy()
        reaction['products'] = products
        reaction['reactants'] = reactants
        reaction['rate_involved'] = rate_involved
        return cls(**reaction)

    def stoichiometry(self):
        return self.multiplicities(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicities(MultiplicityType.rate_involvement)

    def used(self):
        return self.product_species.union(self.reactant_species).union(self.rate_involved_species)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Reaction(name={self.name}, description={self.description}, reactants={self.reactants}, products={self.products}, rate_involvement={self.rate_involved}, k={self.k})"

    def __str__(self) -> str:
        return f"Reaction(name={self.name}, description={self.description}, reactants={self.reactants}, products={self.products}, rate_involvement={self.rate_involved}, k={self.k})"

class RateConstantCluster(NamedTuple):
    k: function
    slice_bottom: int
    slice_top: int

class UnusedSpeciesError(Exception):
    pass

class MissingParametersError(Exception):
    pass

class Model():
    def __init__(self, species: list[Species], reactions: list[Reaction]) -> None:
        if isinstance(reactions, Reaction) or isinstance(reactions, ReactionRateFamily):
            reactions = [reactions]
        if len(reactions) == 0:
            raise ValueError("reactions must include at least one reaction.")
        if isinstance(species, Species):
            species = [species]
        if len(species) == 0:
            raise ValueError("species must include at least one species.")
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
        return {'species': [asdict(s) for s in self.species], 'reaction_groups': [asdict(r) if isinstance(r, Reaction) else r.to_dict() for r in self.reaction_groups]}

    @classmethod
    def from_dict(cls, dictionary, functions_by_name={}):
        dictionary = dictionary.copy()
        species = {}
        for species_dict in dictionary.pop('species'):
            s = Species(**species_dict)
            species[s.name] = s

        reactions = []
        for reaction_dict in dictionary.pop('reaction_groups'):
            # is it a ReactionRateFamily?
            if 'reactions' in reaction_dict.keys():
                try:
                    reactions.append(ReactionRateFamily.from_stringy_species_reactions_k(species, reaction_dict['reactions'], functions_by_name[reaction_dict['k']]))
                except KeyError:
                    raise KeyError(f'failed to find function with name {reaction_dict["k"]} when loading a Model that has a ReactionRateFamily. Did you a pass the keyword argument functions_by_name?')
                continue
            reactions.append(Reaction.from_species_and_dictionary(species, reaction_dict))

        species = [s for s in species.values()]
        return cls(species, reactions, **dictionary)


    def __eq__(self, other: object) -> bool:
        if tuple(self.species) != tuple(other.species):
            return False
        if set(self.all_reactions) != set(other.all_reactions):
            return False
        return True

    def get_k(self, parameters=None, jit=False):
        # ReactionRateFamilies allow us to calculate k(t) for a group of reactions all at once
        base_k = np.zeros(self.n_reactions)
        k_families = []
        i = 0
        # reactions not self.reactions so we see families
        for r in self.reaction_groups:
            # in __init__ we guranteed that one of the following is True:
            # isinstance(r, Reaction) or isinstance(r, ReactionRateFamily)
            if isinstance(r, ReactionRateFamily):
                k_families.append(RateConstantCluster(r.k, i, i+len(r.reactions)+1))
                i += len(r.reactions)
                continue

            # Only reachable if isinstance(r, Reaction)
            assert(isinstance(r,Reaction))
            if isinstance(r.k, str):
                if parameters is None:
                    raise MissingParametersError("attempted to get k(t) without a parameter dictionary where at least one rate constant was a string that needs a parameter dictionary to be evaluated")
                base_k[i] = r.eval_k_with_parameters(parameters)
            elif isinstance(r.k, float):
                base_k[i] = r.k
            elif isinstance(r.k, function) or (jit and isinstance(r.k, CPUDispatcher)):
                k_families.append(RateConstantCluster(r.k, i, i+1))
            else:
                raise TypeError(f"a reaction's rate constant should be, a float, a string expression (evaluated --> float when given parameters), or function with signature k(t) --> float: {r.k}")

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

        # if we have no explicit time dependence, our k function just returns base_k
        if len(k_families) == 0:
            @jit(Array(float64, 1, "C")(float64), nopython=True)
            def k_jit(t):
                k = base_k.copy()
                return k
            return k_jit
        # otherwise, we have to apply the familiy function to differently sized blocks
        k_functions, k_slice_bottoms, k_slice_tops = map(np.array, zip(*k_families))

        if len(k_functions) > 1:
            raise JitNotImplementedError("building a nopython jit function for the rate constants isn't supported with more than 1 subcomponent of k having explicit time dependence. Try using a ReactionRateFamily for all reactions and supplying a vectorized k.")

        # now we have only one subcomponent we have to deal with
        k_function, k_slice_bottom, k_slice_top = k_functions[0], k_slice_bottoms[0], k_slice_tops[0]

        #@jit(Array(float64, 1, "C")(float64), nopython=True)
        @jit(nopython=True)
        def k_jit(t):
            k = base_k.copy()
            k[k_slice_bottom:k_slice_top] = k_function(t)
            return k

        return k_jit

    def multiplicity_matrix(self, mult_type):
        matrix = np.zeros((self.n_species, self.n_reactions))
        for column, reaction in enumerate(self.all_reactions):
            multiplicity_column = np.zeros(self.n_species)
            reaction_info = reaction.multiplicities(mult_type)
            for species, multiplicity in reaction_info.items():
                multiplicity_column[self.species_index[species]] = multiplicity

            matrix[:,column] = multiplicity_column

        return matrix

    def _get_k(self, base_k, k_families):
        def k(t):
            k = base_k.copy()
            for family in k_families:
                k[family.slice_bottom:family.slice_top] = family.k(t)
            return k
        return k

    def stoichiometry(self):
        return self.multiplicity_matrix(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicity_matrix(MultiplicityType.rate_involvement)

    def get_propensities_function(self, jit=False, **kwargs):
        if jit:
            return self._get_jit_propensities_function(**kwargs)
        return self._get_propensities_function(**kwargs)

    def _get_propensities_function(self, parameters=None):
        k_of_t = self.get_k(parameters=parameters, jit=False)
        def calculate_propensities(t, y):
            # product along column in rate involvement matrix
            # with states raised to power of involvement
            # multiplied by rate constants == propensity
            # dimension of y is expanded to make it a column vector
            return np.prod(binom(np.expand_dims(y, axis=1), self.rate_involvement()), axis=0) * k_of_t(t)
        return calculate_propensities

    def _get_jit_propensities_function(self, parameters=None):
        rate_involvement_matrix = self.rate_involvement()
        k_jit = self.get_k(parameters=parameters, jit=True)
        @jit(nopython=True)
        def jit_calculate_propensities(t, y):
            # Remember, we want total number of distinct combinations * k === rate.
            # we want to calculate (y_i rate_involvement_ij) (binomial coefficient)
            # for each species i and each reaction j
            # sadly, inside a numba C function, we can't avail ourselves of scipy's binom,
            # so we write this little calculator ourselves
            intensity_power = np.zeros_like(rate_involvement_matrix)
            for i in range(0, rate_involvement_matrix.shape[0]):
                for j in range(0, rate_involvement_matrix.shape[1]):
                    if y[i] < rate_involvement_matrix[i][j]:
                        intensity_power[i][j] = 0.0
                    elif y[i] == rate_involvement_matrix[i][j]:
                        intensity_power[i][j] = 1.0
                    else:
                        intensity = 1.0
                        for x in range(0, rate_involvement_matrix[i][j]):
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

    def get_dydt_function(self, jit=False, **kwargs):
        if jit:
            return self._get_jit_dydt_function(**kwargs)
        return self._get_dydt_function(**kwargs)

    def _get_dydt_function(self, parameters=None):
        calculate_propensities = self.get_propensities_function(parameters=parameters)
        N = self.stoichiometry()
        def dydt(t, y):
            propensities = calculate_propensities(t, y)

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt = N @ propensities
            return dydt
        return dydt

    def _get_jit_dydt_function(self, parameters=None):
        jit_calculate_propensities = self._get_jit_propensities_function(parameters=parameters)
        N = self.stoichiometry()
        @jit(nopython=True)
        def jit_dydt(t, y):
            propensities = jit_calculate_propensities(t, y)

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt = N @ propensities
            return dydt

        return jit_dydt

    @staticmethod
    def pad_equally_until(string, length, tie='left'):
        missing_length = length - len(string)
        if tie == 'left':
            return " " * int(np.ceil(missing_length/2)) + string + " " * int(np.floor(missing_length/2))
        return " " * int(np.floor(missing_length/2)) + string + " " * int(np.ceil(missing_length/2))

    def pretty_side(self, reaction, side, absentee_value, skip_blanks=False):
        padded_length = 4
        reactant_multiplicities = reaction.multiplicities(side)
        if len(reactant_multiplicities.keys()) == 0:
            return self.pad_equally_until("0", padded_length)

        prior_species_flag = False
        pretty_side = ""
        for i,s in enumerate(self.species):
            mult = reactant_multiplicities.get(s, absentee_value)
            if mult is None:
                species_piece = '' if i==0 or skip_blanks else ' '*2
                pretty_side += species_piece
                if not skip_blanks:
                    pretty_side += " " * padded_length
                prior_species_flag = False
                continue
            if i == 0:
                species_piece = ''
            else:
                species_piece = ' +' if prior_species_flag else ' '*2
            prior_species_flag = True
            species_piece += self.pad_equally_until(f"{str(int(mult)) if mult < 10 else '>9':.2}{s}", padded_length)
            #print(f"piece: |{species_piece}|")
            pretty_side += species_piece
        return pretty_side

    def pretty(self, hide_absent=True, skip_blanks=True, max_width=120) -> str:
        absentee_value = None if hide_absent else 0
        pretty = ""
        for reaction in self.all_reactions:
            pretty_reaction = f"{reaction.description:.22}" + " " * max(0, 22-len(reaction.description)) + ":"
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.reacants, absentee_value, skip_blanks)
            pretty_reaction += ' --> '
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.products, absentee_value, skip_blanks)

            pretty += pretty_reaction + '\n'
        return pretty

    def make_initial_condition(self, dictionary):
        x0 = np.zeros(self.n_species)
        for k,v in dictionary.items():
            x0[self.species_name_index[k]] = float(v)
        return x0

class JitNotImplementedError(Exception):
    pass

@dataclass(frozen=True)
class ReactionRateFamily():
    reactions: list[Reaction]
    k: callable

    def used(self):
        used = set()
        for r in self.reactions:
            for s in r.used():
                used.add(s)
        return used

    def to_dict(self):
        return {'reactions': [asdict(r) for r in self.reactions], 'k': self.k.__name__}

    @classmethod
    def from_stringy_species_reactions_k(cls, species, reactions, k):
        self_dict = {}
        our_reactions = []
        for r_dict in reactions:
            our_reactions.append(Reaction.from_species_and_dictionary(species, r_dict))

        self_dict['reactions'] = our_reactions
        self_dict['k'] = k

        return cls(**self_dict)
