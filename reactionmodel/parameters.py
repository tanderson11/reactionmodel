from dataclasses import dataclass, make_dataclass
import numpy as np

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
class Parametrization():
    pass
    
    def todict(self):
        d = {}
        for k,v in self.__dict__.items():
            if isinstance(v, ImmutableArray):
                d[k] = v.to_np_array()
            else:
                d[k] = v
        return d

def make_parametrization(parameters):
    fields = []
    typed_parameters = {}

    for k,v in parameters.items():
        if isinstance(v, np.ndarray):
            fields.append((k, ImmutableArray))
            typed_parameters[k] = ImmutableArray.from_np_array(v)
        else:
            fields.append((k, float))
            typed_parameters[k] = float(v)
        
    klass = make_dataclass('ModelParametrization', fields, bases=(Parametrization,), frozen=True)
    return klass(**typed_parameters)