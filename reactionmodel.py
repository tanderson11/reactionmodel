import numpy as np
from enum import Enum
from typing import NamedTuple
from types import FunctionType as function
from simpleeval import simple_eval
from dataclasses import dataclass, asdict

NO_NUMBA = False
try:
    from numba import jit, float64
    from numba.types import Array
except ModuleNotFoundError:
    NO_NUMBA = True

class Species():
    def __init__(self, name, abbreviation) -> None:
        self.name = name
        self.abbreviation = abbreviation

    def __str__(self):
        return self.abbreviation

    def __hash__(self) -> int:
        return hash((self.name, self.abbreviation))

    def __repr__(self) -> str:
        return f"Species(name={self.name}, abbreviation={self.abbreviation})"

class MultiplicityType(Enum):
    reacants = 'reactants'
    products = 'products'
    stoichiometry = 'stoichiometry'
    rate_involvement = 'rate involvement'

class Reaction():
    def __init__(self, description, reactants, products, rate_involvement=None, k=None, reversible=False) -> None:
        assert reversible == False, "Reversible reactions are not supported. Create separate forward and back reactions instead."
        self.description = description

        self.k = k

        if isinstance(reactants, Species) or isinstance(reactants, tuple):
            reactants = [reactants]
        if isinstance(products, Species) or isinstance(products, tuple):
            products = [products]
        self.reactants = set([(r[0] if isinstance(r, tuple) else r) for r in reactants])
        self.products = set([(p[0] if isinstance(p, tuple) else p) for p in products])
        self.reactant_data = reactants
        self.product_data = products

        self.rate_involvement = self.reactants if rate_involvement is None else rate_involvement

    def eval_k_with_parameters(self, parameters):
        if not isinstance(parameters, dict):
            parameters = asdict(parameters)
        k = simple_eval(self.k, names=parameters)
        if not isinstance(k, float):
            raise ValueError(f"Evaluation of Reaction k defined by the string (s) did not produce a float literal", self.k)
        return k

    def multiplicities(self, mult_type):
        multplicities = {}

        positive_multplicity_data = []
        negative_multiplicity_data = []
        if mult_type == MultiplicityType.reacants:
            positive_multplicity_data = self.reactant_data
        elif mult_type == MultiplicityType.products:
            positive_multplicity_data = self.product_data
        elif mult_type == MultiplicityType.stoichiometry:
            positive_multplicity_data = self.product_data
            negative_multiplicity_data = self.reactant_data
        elif mult_type == MultiplicityType.rate_involvement:
            positive_multplicity_data = self.rate_involvement
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

    def stoichiometry(self):
        return self.multiplicities(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicities(MultiplicityType.rate_involvement)

    def __repr__(self) -> str:
        return f"Reaction(description={self.description}, reactants={self.reactant_data}, products={self.product_data}, rate_involvement={self.rate_involvement}, k={self.k})"

class RateConstantCluster(NamedTuple):
    k: function
    slice_bottom: int
    slice_top: int

class Model():
    def __init__(self, species: list[Species], reactions: list[Reaction], parameters=None, jit=False) -> None:
        if isinstance(reactions, Reaction) or isinstance(reactions, ReactionRateFamily):
            reactions = [reactions]
        if len(reactions) == 0:
            raise ValueError("reactions must include at least one reaction.")
        if len(species) == 0:
            raise ValueError("species must include at least one species.")
        self.species = species
        self.reactions = []
        for r in reactions:
            if isinstance(r, Reaction):
                self.reactions.append(r)
            elif isinstance(r, ReactionRateFamily):
                self.reactions.extend(r.reactions)
            else:
                raise TypeError(f"bad type for reaction in model: {type(r)}. Expected Reaction or ReactionRateFamily")

        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)

        self.species_index = {s:i for i,s in enumerate(self.species)}
        self.reaction_index = {r:i for i,r in enumerate(self.reactions)}

        self.jit = jit
        # If the rate constants are specified in a "lazy" way that depends on receiving a Parameters object in the future,
        # we lock some methods of this class until the rate constants have been "baked" properly
        self.k_lock = self.bake_k(parameters=parameters)


    def bake_k(self, parameters=None):
        # ReactionRateFamilies allow us to calculate k(t) for a group of reactions all at once
        base_k = np.zeros(self.n_reactions)
        k_of_ts = []
        i = 0
        # reactions not self.reactions so we see families
        for r in self.reactions:
            # in __init__ we guranteed that one of the following is True:
            # isinstance(r, Reaction) or isinstance(r, ReactionRateFamily)
            if isinstance(r, ReactionRateFamily):
                k_of_ts.append(RateConstantCluster(r.k, i, i+len(r.reactions)+1))
                i += len(r.reactions)
                continue

            # Only reachable if isinstance(r, Reaction)
            assert(isinstance(r,Reaction))
            if isinstance(r.k, str):
                if parameters is None:
                    k_lock =  True
                    notice = "NOTICE: At least one reaction rate constant was a string, but no parameters were provided to decode it. Calculating k(t) will be disabled until Model.bake_k(parameters=parameters) is run."
                    print(notice)
                    if self.jit:
                        self.k_jit = notice
                    return k_lock
                base_k[i] = r.eval_k_with_parameters(parameters)
            elif isinstance(r.k, float):
                base_k[i] = r.k
            elif isinstance(r.k, function):
                k_of_ts.append(RateConstantCluster(r.k, i, i+1))
            else:
                raise TypeError(f"a reaction's rate constant should be a float or function with signature k(t) --> float: {r.k}")

            i+=1

        self.base_k = base_k
        self.k_of_ts = k_of_ts

        k_lock = False
        return k_lock

        if self.jit:
            if NO_NUMBA:
                raise ModuleNotFoundError("""No module named 'numba'. To use jit=True functions, you must install this package with extras. Try `poetry add "reactionmodel[extras]"` or `pip install "reactionmodel[extras]".""")
            # convert families into relevant lists
            self.k_jit = self.kjit_factory(np.array(self.base_k), self.k_of_ts)

    def kjit_factory(self, base_k, k_families):
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
        k_functions, k_slice_bottoms, k_slice_tops = map(np.array, zip(*self.k_of_ts))

        if len(k_functions) > 1:
            raise JitNotImplementedError("building a nopython jit for the rate constants isn't supported with more than 1 subcomponent of k having explicit time dependence. Try using a ReactionRateFamily for all reactions and supplying a vectorized k.")

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
        for column, reaction in enumerate(self.reactions):
            multiplicity_column = np.zeros(self.n_species)
            reaction_info = reaction.multiplicities(mult_type)
            for species, multiplicity in reaction_info.items():
                multiplicity_column[self.species_index[species]] = multiplicity

            matrix[:,column] = multiplicity_column

        return matrix

    @staticmethod
    def check_k_lock(self):
        if self.k_lock():
            raise AttributeError("attempted to evaluate k for a Model before rate constants were evaluated. Try `Model.bake_k(parameters=parameters)` first.")

    def k(self, t):
        self.check_k_lock()
        k = self.base_k.copy()
        for family in self.k_of_ts:
            k[family.slice_bottom:family.slice_top] = family.k(t)
        return k

    def stoichiometry(self):
        return self.multiplicity_matrix(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicity_matrix(MultiplicityType.rate_involvement)

    def get_propensities_function(self):
        def calculate_propensities(t, y):
            # product along column in rate involvement matrix
            # with states raised to power of involvement
            # multiplied by rate constants == propensity
            # dimension of y is expanded to make it a column vector
            return np.prod(np.expand_dims(y, axis=1)**self.rate_involvement(), axis=0) * self.k(t)
        return calculate_propensities

    def get_dydt_function(self):
        calculate_propensities = self.get_propensities_function()
        N = self.stoichiometry()
        def dydt(t, y):
            propensities = calculate_propensities(t, y)

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt = N @ propensities
            return dydt
        return dydt

    def get_jit_propensities_function(self):
        try:
            self.k_jit
        except AttributeError:
            assert False, "Numba JIT functions may only be acquired if Model was created with jit=True"

        @jit(nopython=True)
        def jit_calculate_propensities(t, y):
            # product along column in rate involvement matrix
            # with states raised to power of involvement
            # multiplied by rate constants == propensity
            # dimension of y is expanded to make it a column vector
            intensity_power = np.expand_dims(y, axis=1)**rate_involvement_matrix
            k = self.k_jit(t)
            product_down_columns = np.ones(len(k))
            for i in range(0, len(y)):
                product_down_columns = product_down_columns * intensity_power[i]
            return product_down_columns * k
        return jit_calculate_propensities

    def get_jit_dydt_function(self):
        try:
            self.k_jit
        except AttributeError:
            assert False, "Numba JIT functions may only be acquired if Model was created with jit=True"

        jit_calculate_propensities = self.get_jit_propensities_function()
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
            species_piece += self.pad_equally_until(f"{str(int(mult)) if mult < 10 else '>9':.2}{s:.2}", padded_length)
            #print(f"piece: |{species_piece}|")
            pretty_side += species_piece
        return pretty_side

    def pretty(self, hide_absent=True, skip_blanks=False) -> str:
        absentee_value = None if hide_absent else 0
        pretty = ""
        for reaction in self.reactions:
            pretty_reaction = f"{reaction.description:.22}" + " " * max(0, 22-len(reaction.description)) + ":"
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.reacants, absentee_value, skip_blanks)
            pretty_reaction += ' --> '
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.products, absentee_value, skip_blanks)

            pretty += pretty_reaction + '\n'
        return pretty

class JitNotImplementedError(Exception):
    pass

class ReactionRateFamily():
    def __init__(self, reactions, k) -> None:
        self.reactions = reactions
        self.k = k