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
class FrozenDictionary():
    subclass_name = 'SpecificFrozenDict'
    pass

    @classmethod
    def make(cls, dictionary):
        fields = []
        typed_parameters = {}

        for k,v in dictionary.items():
            field_type, field_value = cls.handle_field(k, v)
            fields.append((k, field_type))
            typed_parameters[k] = field_value
            
        klass = make_dataclass(cls.subclass_name, fields, bases=(cls,), frozen=True)
        return klass(**typed_parameters)

    @staticmethod
    def handle_field(k, v):
        return (type(v),v)

    def asdict(self):
        d = {}
        for k,v in self.__dict__.items():
            d[k] = v
        return d