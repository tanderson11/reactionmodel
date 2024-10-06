from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Species, RatelessReaction, Reaction, Model, MultiplicityType, eval_expression

@dataclass(frozen=True)
class Hook():
    """A Hook has a stoichiometry matrix and a probability vector.
    
    Upon a triggering condition, an index is selected from the probability vector and the resultant update vector is provided."""
    N: np.ndarray
    p: np.ndarray

    def __post_init__(self):
        assert np.isclose(self.p.sum(), 1), "total probability of the hook must be 1"

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
        for r in self.hooked_reactions:
            used = used.union(r.used())
        return used

class HookAwareModel(Model):
    def __init__(self, species: list[Species], reactions: list[Reaction], reject_duplicates=True) -> None:
        super().__init__(species, reactions, reject_duplicates)

        # TK make sure this works with reaction groups
        self.reaction_index_to_hooks = {}
        for i,r in enumerate(self.all_reactions):
            if not isinstance(r, ReactionWithHooks):
                continue
            if r.hooked_reactions is None:
                continue
            self.reaction_index_to_hooks[i] = self.build_hook(r.hooked_reactions, r.hooked_p)
    
    def build_hook(self, reactions: list[RatelessReaction], p):
        N = self.multiplicity_matrix(MultiplicityType.stoichiometry, reactions)
        return Hook(N, np.array(p))