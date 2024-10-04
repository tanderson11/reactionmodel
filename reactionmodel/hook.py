from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Species, RatelessReaction, Reaction, Model, MultiplicityType, eval_expression

@dataclass(frozen=True)
class HookedReactions():
    """HookedReactions specifies a set of reactions, 1 of which occurs upon a triggering condition."""
    reactions: list[RatelessReaction]
    p: np.ndarray

    def __post_init__(self):
        assert self.p.shape == np.array(self.reactions).shape, "probability vector "
        assert isinstance(self.p, (np.floating, float, str)), "probability of hooks must be floats or lazy strings"

    def eval_k_with_parameters(self, parameters):
        """Evaluate lazy rate constant in the context of parameters."""
        k = eval_expression(self.k, parameters)
        return k

@dataclass(frozen=True)
class Hook():
    """A Hook has a stoichiometry matrix and a probability vector.
    
    Upon a triggering condition, an index is selected from the probability vector and the resultant update vector is provided."""
    N: np.ndarray
    p: np.ndarray

    def __post_init__(self):
        assert np.isclose(p.sum(), 1), "total probability of the hook must be 1"

@dataclass(frozen=True)
class ReactionWithHooks(Reaction):
    hooked_reactions: HookedReactions = None

class HookAwareModel(Model):
    def __init__(self, species: list[Species], reactions: list[Reaction], reject_duplicates=True) -> None:
        super().__init__(species, reactions, reject_duplicates)

        # TK make sure this works with reaction groups
        self.reaction_index_to_hooks = {}
        for i,r in enumerate(self.all_reactions):
            if not isinstance(r, ReactionWithHooks):
                continue
            if r.ReactionWithHooks is None:
                continue
            self.reaction_index_to_hooks[i] = self.make_hook_from_hook_set(r.hooked_reactions.reactions, r.hooked_reactions.p)
    
    def build_hook(self, reactions: list[RatelessReaction], p):
        N = self.multiplicity_matrix(MultiplicityType.stoichiometry, reactions)
        return Hook(N, p)