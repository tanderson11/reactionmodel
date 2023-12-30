from dataclasses import dataclass
import numpy as np
from reactionmodel.util import FrozenDictionary
from reactionmodel.util import ImmutableArray

@dataclass(frozen=True, eq=False)
class Parametrization(FrozenDictionary):
    subclass_name = 'SpecifiedParameters'

    def asdict(self, rebuild_arrays=True):
        d = {}
        for k,v in self.__dict__.items():
            if rebuild_arrays and isinstance(v, ImmutableArray):
                d[k] = v.to_np_array()
            else:
                d[k] = v
        return d

    @staticmethod
    def handle_field(k, v):
        if isinstance(v, np.ndarray):
            return (ImmutableArray, ImmutableArray.from_np_array(v))
        else:
            return (float, float(v))

    def __eq__(self, other: object) -> bool:
        return self.asdict(rebuild_arrays=False) == other.asdict(rebuild_arrays=False)