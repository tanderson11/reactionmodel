import re
from reactionmodel import Species, Reaction, Model
from dataclasses import dataclass
from itertools import product

# model specification language

## Bad constraints:
### overlapping family names like x and xx will go wild

class AtomDecoder():
    klass = None
    properties = []
    optional_properties = []

    @classmethod
    def decode_property(cls, name, value):
        return value

    @classmethod
    def decode(cls, name, properties, existing_atoms={}):
        ps = []
        optional_ps = {}

        required_properties = [p.name for p in cls.properties if not p.optional]
        optional_properties = [p.name for p in cls.properties if p.optional]

        # go through required properties IN ORDER
        for p in required_properties:
            try:
                v = properties.pop(p)
            except KeyError:
                raise MissingRequiredPropertyError(f'{name} is missing {p}')
            ps.append(v.value)
        for p,v in properties.items():
            if p not in optional_properties:
                raise UnexpectedPropertyError(f'{name} had unexpected property {p}')
            optional_ps[p] = v.value

        return cls.klass(name, *ps, **optional_ps)

class PropertyMatch():
    def __init__(self, property_name, value):
        self.property_name = property_name
        self.value = value

    @staticmethod
    def localize_string_with_family_members(string, syntax, families, chosen_members):
        for family_name, member in zip(families, chosen_members):
            string = string.replace(syntax.family_denoter + family_name, syntax.family_denoter + member)
        return string

    def localize_with_family_members(self, syntax, families, chosen_members):
        return self.localize_string_with_family_members(self.value, syntax, families, chosen_members)

class ListMatch(PropertyMatch):
    def localize_with_family_members(self, syntax, families, chosen_members):
        new_value = [self.localize_string_with_family_members(v, syntax, families, chosen_members) for v in new_value]
        return new_value

class Property():
    value_pattern_string = '([a-zA-Z0-9]+)$'
    match_klass = PropertyMatch
    def __init__(self, name, optional=False, alternative=None) -> None:
        self.name = name
        self.optional = optional
        self.alternative = alternative
        self.compiled = None

    def __hash__(self) -> int:
        return hash(self.name)

    def get_pattern(self, syntax):
        if self.compiled is None:
            self.compiled = re.compile(f'^({self.name}): ' + self.inject_syntax(syntax))
        return self.compiled

    def inject_syntax(self, syntax):
        return self.value_pattern_string

    def parse(self, line, syntax):
        pattern = self.get_pattern(syntax)
        match = re.match(pattern, line)
        if match is not None:
            return self.match_klass(match[1], match[2])
        return None

class RichProperty(Property):
    value_pattern_string = '"(.*)"$'

class PathProperty(Property):
    value_pattern_string = '"(.*)"$'

class ReactionProperty(Property):
    # digits, reaction arrow, family denoter, otherwise same things that can be in a name
    value_pattern_string = '(.*)$'
    species_pattern = re.compile('^([0-9]*)(.*?)$')

    def inject_syntax(self, syntax):
        return self.value_pattern_string.format(syntax.reaction_arrow, syntax.family_denoter)

    def parse(self, line, syntax):
        match = super().parse(line, syntax)
        left_side, right_side = match.value.split(syntax.reaction_arrow)
        for side in (left_side, right_side):
            side = side.split(' ')
            for species_string in side:
                species_match = re.match(self.species_pattern, species_string)
                if species_match is None:
                    raise BadSpeciesInReactionError(f"couldn't understand {species_string} as a [multiplicity]SpeciesName")
                # species multiplicity
                multiplicity = species_match[1]
                species = species_match[2]
        # DO SOMETHING TK TK TK

class ListProperty(Property):
    value_pattern_string = '([a-zA-Z0-9{0}]+)$'
    match_klass = ListMatch

    def inject_syntax(self, syntax):
        return self.value_pattern_string.format(syntax.list_delimiter)

    def parse(self, line, syntax):
        property_match = super().parse(line, syntax)
        # split the list!
        if property_match is not None:
            property_match.value = property_match.value.split(syntax.list_delimiter)
            return property_match

        return None

class Family():
    def __init__(self, name, members, description=""):
        self.name = name
        self.members = members
        self.description = description

    def __repr__(self) -> str:
        return f"Family {self.name}: with members={self.members}"

class FamilyDecoder(AtomDecoder):
    klass = Family
    header = "Family"
    properties = [ListProperty('members'), Property('description', optional=True)]

class SpeciesDecoder(AtomDecoder):
    klass = Species
    header = "Species"
    properties = [RichProperty('description', optional=True)]

class ReactionDecoder(AtomDecoder):
    klass = Reaction
    header = "Reaction"
    properties = []

class ModelSyntaxError(Exception):
    pass

class MissingRequiredPropertyError(Exception):
    pass

class UnexpectedPropertyError(Exception):
    pass

class DuplicateAtomNameError(Exception):
    pass

class BadSpeciesInReactionError(Exception):
    pass

@dataclass(frozen=True)
class Syntax():
    atom_separator = '\n'
    family_denoter = '_'
    list_delimiter = ','
    colon_equivalent = ':'
    period_equivalent = '.'
    reaction_arrow = '->'

class Parser():
    decoders = [SpeciesDecoder, FamilyDecoder]

    def __init__(self, syntax=Syntax(), decoders=None) -> None:
        self.syntax = syntax
        self.position_pattern = re.compile(f'^( +)?(.*?)([{syntax.period_equivalent}{syntax.colon_equivalent}])?$')
        self.header_pattern = re.compile(f'^([a-zA-Z{syntax.family_denoter}]+) ([a-zA-Z0-9_]+)$')
        if decoders:
            self.decoders = decoders
        self.decoder_lookup = {d.header: d for d in self.decoders}

    def localize_properties_with_family_members(self, atom_properties, family_names, member_choices):
        # member choices == list of ordered pairs (family name, which member)
        new_dictionary = atom_properties.copy()
        for parameter_name, parameter in new_dictionary.items():
            new_dictionary[parameter_name] = parameter.localize_with_family_members(self.syntax, family_names, member_choices)
        return new_dictionary

    def add_atoms(self, existing_atoms, decoder, atom_name, atom_properties, i):
        new_atoms = {}
        if self.syntax.family_denoter in atom_name:
            _, *families = atom_name.split(self.syntax.family_denoter)
            family_members = []
            for family_name in families:
                family = existing_atoms.get(family_name, None)
                if family is None or not(isinstance(family, Family)):
                    raise ModelSyntaxError(f"looked for family {family_name} but couldn't find its definition. L {i+1}")
                # append the *list*, so we can keep our families straight
                family_members.append(family.members)
            for combination in product(*family_members):
                print(combination)
                new_atoms[PropertyMatch.localize_string_with_family_members(atom_name, self.syntax, families, combination)] = self.localize_properties_with_family_members(atom_properties, families, combination)
        else:
            new_atoms[atom_name] = self.decode_atom(existing_atoms, decoder, atom_name, atom_properties, i)
        existing_atoms.update(new_atoms)
        return existing_atoms

    def decode_atom(self, atoms, decoder, atom_name, atom_properties, i):
        if atoms.get(atom_name, None) is not None:
            raise DuplicateAtomNameError(f"Duplicate atom name {atom_name}. L:{i+1}")
        return decoder.decode(atom_name, atom_properties, atoms)

    def parse_file(self, file):
        with open(file, 'r') as f:
            raw = f.readlines()
        return self.parse_lines(raw)

    def parse_lines(self, lines):
        expect_header = True
        expect_blank = False

        i = 0
        atoms = {}
        atom_properties = {}
        atom_header = None
        atom_decoder = None
        atom_name = None
        whitespace = None
        for i,l in enumerate(lines):
            #print(l, l=='', l=='\n', l=='\r')

            postion_match = re.match(self.position_pattern, l)
            self.check_match(postion_match, l, i+1, "a valid line")
            indent, line_body, terminator = postion_match[1], postion_match[2], postion_match[3]
            #print(i)
            #print(indent, line_body, terminator)

            # check if indent is acceptable
            if indent and expect_header:
                raise ModelSyntaxError(f"Unexpected indent. L:{i+1}")

            # if blank line, verify that is acceptable and then pipe all previous atom lines together
            if l==self.syntax.atom_separator:
                if not expect_blank:
                    raise ModelSyntaxError(f"Unexpected blank line. L:{i+1}. Did you terminate this unit with a {self.syntax.period_equivalent}?")

                # build (potentially several -- if family) new atoms from name, properties, and all the existing atoms
                atoms = self.add_atoms(atoms, atom_decoder, atom_name, atom_properties, i)

                expect_blank = False
                expect_header = True
                atom_properties = {}
                atom_header = None
                atom_name = None
                atom_decoder = None
                continue

            if expect_header:
                match = re.match(self.header_pattern, line_body)
                self.check_match(match, line_body, i+1, 'ObjectType Name:')
                atom_header = match[1]
                atom_name = match[2]
                try:
                    atom_decoder = self.decoder_lookup[atom_header]
                except KeyError:
                    raise ModelSyntaxError(f"No decoder found for {atom_header}. L:{i+1}")
                expect_header = False
                expect_blank = (terminator == self.syntax.period_equivalent)
                continue

            if whitespace is None:
                whitespace = indent
            if whitespace != indent:
                raise ModelSyntaxError(f'Inconsistent whitespace before property: value pair: "{l}" on L{i+1}')

            for property_parser in atom_decoder.properties:
                match = property_parser.parse(line_body, self.syntax)
                if match is not None:
                    break

            self.check_match(match, line_body, i+1, f'keyword: value (for the predefined properties of {atom_header})')
            atom_properties[match.property_name] = match
            expect_blank = (terminator == self.syntax.period_equivalent)
            i+=1

        # for loop over; we are out of lines
        if not expect_blank:
            raise ModelSyntaxError(f'ran out of lines while parsing a single unit. Every Species/Reaction/Model should have its last line terminated by a {self.syntax.period_equivalent}')

        # make our last atom!
        atoms = self.add_atoms(atoms, atom_decoder, atom_name, atom_properties, i)

        print(atoms)
        return atoms

    @staticmethod
    def check_match(match, line, i, expected):
        if match is None:
            raise ModelSyntaxError(f'expected to find a {expected} but found "{line}" on L:{i}')

if __name__ == '__main__':
    import sys
    p = Parser()
    p.parse_file(sys.argv[1])