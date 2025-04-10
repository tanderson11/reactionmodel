from dataclasses import dataclass
import numpy as np
import reactionmodel.syntax
from reactionmodel.model import Species, RatelessReaction, Reaction, Model, MultiplicityType, eval_expression

class BadHookProbabilityException(Exception):
    pass

@dataclass(frozen=True)
class Hook():
    """A Hook has a stoichiometry matrix and a probability vector.

    Upon a triggering condition, an index is selected from the probability vector and the resultant update vector is provided."""
    N: np.ndarray
    p: np.ndarray

    def __post_init__(self):
        if not np.isclose(self.p.sum(), 1):
            raise(BadHookProbabilityException( f"total probability of the hook must be 1, is {self.p.sum()}"))

@dataclass(frozen=True)
class ReactionWithHooks(Reaction):
    hooked_reactions: tuple[RatelessReaction] = None
    hooked_p: tuple[float, str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.hooked_reactions is None or self.hooked_p is None:
            assert self.hooked_reactions is None and self.hooked_p is None
            return

        if not isinstance(self.hooked_reactions, tuple):
            object.__setattr__(self, 'hooked_reactions', tuple(self.hooked_reactions))
        if not isinstance(self.hooked_p, tuple):
            object.__setattr__(self, 'hooked_p', tuple(self.hooked_p))

        p_array = np.array(self.hooked_p)
        assert np.array(p_array).shape == np.array(self.hooked_reactions).shape, "probability vector should have same shape as hooked reactions"
        assert np.issubdtype(p_array.dtype, np.floating) or np.issubdtype(p_array.dtype, str), "probability of hooks must be floats or lazy strings"

    def used(self):
        used = super().used()
        if self.hooked_reactions is None:
            return used
        for r in self.hooked_reactions:
            used = used.union(r.used())
        return used

    def eval_p_with_parameters(self, parameters):
        """Evaluate lazy rate constant in the context of parameters."""
        p = [eval_expression(_p, parameters) for _p in self.hooked_p]
        p = np.array(p)
        return p

    @classmethod
    def _preprocess_dictionary_for_loading(cls, dictionary, species_context, reaction_class=RatelessReaction):
        self_dict = super()._preprocess_dictionary_for_loading(dictionary, species_context)

        # now we need to make all the inner reaction dictionaries into actual reactions
        hooked_reactions = dictionary.pop('hooked_reactions', None)
        if hooked_reactions is None:
            return self_dict
        our_reactions = []
        for r_dict in hooked_reactions:
            r_dict = r_dict.copy()
            r_dict.pop('p')
            our_reactions.append(reaction_class.from_dict(r_dict, species_context))

        self_dict['hooked_reactions'] = our_reactions
        return self_dict

    @classmethod
    def from_dict(cls, dictionary, species_context):
        """Given dictionary representation of a Reaction and a species_context of Species objects, return a Reaction"""
        reaction_dict = cls._preprocess_dictionary_for_loading(dictionary, species_context)

        return cls(**reaction_dict)

    def to_dict(self, keep_species_objects=False):
        selfdict = super().to_dict(keep_species_objects)
        selfdict['hooked_reactions'] = self.hooked_reactions
        selfdict['hooked_p'] = self.hooked_p
        return selfdict

class HookAwareModel(Model):
    has_hooks=True
    def __init__(self, species: list[Species], reactions: list[Reaction], reject_duplicates=True) -> None:
        super().__init__(species, reactions, reject_duplicates)

    def build_hook(self, reactions: list[RatelessReaction], p):
        N = self.multiplicity_matrix(MultiplicityType.stoichiometry, reactions)
        return Hook(N, np.array(p))

    def get_hooks(self, parameters: dict=None):
        if parameters is None:
            parameters = {}
        # TK make sure this works with reaction groups
        reaction_index_to_hooks = {}
        for i,r in enumerate(self.all_reactions):
            if not isinstance(r, ReactionWithHooks):
                continue
            if r.hooked_reactions is None:
                continue
            p = r.eval_p_with_parameters(parameters=parameters)
            #try:
            reaction_index_to_hooks[i] = self.build_hook(r.hooked_reactions, p)
            #except BadHookProbabilityException as e:
            #    raise BadHookProbabilityException(str(r))

        return reaction_index_to_hooks

    @classmethod
    def parse_model(cls, families, species, reactions, syntax=reactionmodel.syntax.Syntax(), triggered_sets=None, **kwargs):
        parsed_triggered_sets = {}
        parsed_triggered_ps = {}
        for k,s in triggered_sets.items():
            #p = np.zeros(len(s))
            triggered_reactions = []
            for r in s:
                triggered_reactions.extend(syntax.expand_families(families, r))
            parsed_triggered_sets[k] = triggered_reactions
            #p = [r.pop('p') for r in triggered_reactions]
            p = [r['p'] for r in triggered_reactions]
            parsed_triggered_ps[k] = p

        all_species = []
        for s in species:
            all_species.extend(syntax.expand_families(families, s))
        all_reactions = []
        for r in reactions:
            all_reactions.extend(syntax.expand_families(families, r))

        for r in all_reactions:
            hooked_set = r.pop('hooked_set', None)
            if hooked_set is None:
                continue
            r['hooked_reactions'] = parsed_triggered_sets[hooked_set]
            r['hooked_p'] = parsed_triggered_ps[hooked_set]

        model_dict = {
            'species'  : all_species,
            'reactions': all_reactions,
        }
        return cls.from_dict(model_dict, **kwargs, reaction_class=ReactionWithHooks)
