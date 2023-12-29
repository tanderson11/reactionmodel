from dataclasses import dataclass
import numpy as np
from reactionmodel.util import FrozenDictionary

@dataclass(frozen=True)
class ImmutableArray():
    data: tuple
    shape: tuple

    @classmethod
    def from_np_array(cls, np_array) -> None:
        return cls(tuple(np_array.flatten().tolist()), np_array.shape)

    def to_np_array(self):
        return np.reshape(np.array(self.data), self.shape)

@dataclass(frozen=True)
class Parametrization(FrozenDictionary):
    subclass_name = 'SpecifiedParameters'

    def asdict(self):
        d = {}
        for k,v in self.__dict__.items():
            if isinstance(v, ImmutableArray):
                d[k] = v.to_np_array()
            else:
                d[k] = v
        return d

    @staticmethod
    def handle_field(k, v):
        if isinstance(v, np.ndarray):
            return (ImmutableArray, ImmutableArray.from_np_array(v))
        else:
            return (FloatingPointError, float(v))