import re
from reactionmodel import Species, Reaction, Model
from dataclasses import dataclass
from itertools import product

# model specification language

## Constraints:
### overlapping family names like x and xx might go wild
### family names must be only alpha

## Wishlist:
### Wait for full evaluation, so some things can be provided later by the user?

class PropertyMatch():
    def __init__(self, property_name, value):
        self.property_name = property_name
        self.value = value

    @staticmethod
    def localize_string_with_family_members(string, syntax, families, chosen_members):
        for family_name, member in zip(families, chosen_members):
            string = string.replace(syntax.family_denoter + family_name, syntax.family_denoter + member)
        return string

    def localize_value_with_family_members(self, syntax, families, chosen_members):
        return self.localize_string_with_family_members(self.value, syntax, families, chosen_members)

    def localize_with_family_members(self, syntax, families, chosen_members):
        value = self.localize_value_with_family_members(syntax, families, chosen_members)
        return self.__class__(self.property_name, value)

    def evaluate_with_existing_atoms(self, existing_atoms):
        return self.value

class ListMatch(PropertyMatch):
    def localize_value_with_family_members(self, syntax, families, chosen_members):
        new_value = [self.localize_string_with_family_members(v, syntax, families, chosen_members) for v in self.value]
        return new_value

class SpeciesMultiplicityListMatch(ListMatch):
    def localize_value_with_family_members(self, syntax, families, chosen_members):
        print(self.value)
        new_value = []
        for v in self.value:
            if isinstance(v, tuple):
                new_value.append((self.localize_string_with_family_members(v[0], syntax, families, chosen_members), v[1]))
            else:
                new_value.append(self.localize_string_with_family_members(v, syntax, families, chosen_members))
        print(new_value)
        return new_value

    def evaluate_with_existing_atoms(self, existing_atoms):
        actual_list = []
        for v in self.value:
            # tuple like (species_name, multiplicity)
            if isinstance(v, tuple):
                actual_list.append((existing_atoms[v[0]], v[1]))
            else:
                actual_list.append(existing_atoms[v])
        return actual_list

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
        print(pattern)
        match = re.match(pattern, line)
        if match is not None:
            return self.match_klass(match[1], match[2])
        return None

class RichProperty(Property):
    value_pattern_string = '"(.*)"$'

class PathProperty(Property):
    value_pattern_string = '"(.*)"$'

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

class SpeciesMultiplicityListProperty(ListProperty):
    match_klass = SpeciesMultiplicityListMatch
    species_pattern = re.compile('^([0-9]*)(.*?)$')
    value_pattern_string = '([a-zA-Z0-9{0}{1}]+)$'

    def inject_syntax(self, syntax):
        return self.value_pattern_string.format(syntax.list_delimiter, syntax.family_denoter)

    def parse(self, line, syntax):
        # gets split to a list by superclass
        # now we format the list as tuples of (species, multplicity) or just species if multiplicity = 1
        property_match = super().parse(line, syntax)
        if property_match is None:
            return property_match

        species_multiplicity_list = []
        for species_string in property_match.value:
            species_match = re.match(self.species_pattern, species_string)
            if species_match is None:
                raise BadSpeciesInReactionError(f"couldn't understand {species_string} as a [multiplicity]SpeciesName.")
            multiplicity = species_match[1]
            species = species_match[2]
            if multiplicity:
                species_multiplicity_list.append((species, int(multiplicity)))
            else:
                species_multiplicity_list.append(species)

        property_match.value = species_multiplicity_list
        return property_match

class AtomFactory():
    klass = None
    properties = []
    optional_properties = []

    @classmethod
    def construct(cls, name, property_matches, existing_atoms={}):
        ps = []
        optional_ps = {}

        required_properties = [p.name for p in cls.properties if not p.optional]
        optional_properties = [p.name for p in cls.properties if p.optional]

        # go through required properties IN ORDER
        for p in required_properties:
            try:
                v = property_matches.pop(p)
            except KeyError:
                raise MissingRequiredPropertyError(f'{name} is missing {p}.')
            ps.append(v.evaluate_with_existing_atoms(existing_atoms))
        for p,v in property_matches.items():
            if p not in optional_properties:
                raise UnexpectedPropertyError(f'{name} had unexpected property {p}.')
            optional_ps[p] = v.evaluate_with_existing_atoms(existing_atoms)

        return cls.klass(name, *ps, **optional_ps)

class Family():
    def __init__(self, name, members, description=""):
        self.name = name
        self.members = members
        if members is None:
            raise ValueError(f"family {name} has None members.")
        self.description = description

    def __repr__(self) -> str:
        return f"Family {self.name}: with members={self.members}"

class FamilyFactory(AtomFactory):
    klass = Family
    header = "Family"
    properties = [ListProperty('members'), Property('description', optional=True)]

class SpeciesFactory(AtomFactory):
    klass = Species
    header = "Species"
    properties = [RichProperty('description', optional=True)]

class ReactionFactory(AtomFactory):
    # name, description, reactants, products, rate_involvement=None, k=None, reversible=False
    klass = Reaction
    header = "Reaction"
    properties = [SpeciesMultiplicityListProperty('reactants'), SpeciesMultiplicityListProperty('products'), RichProperty('description', optional=True)]

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
    factories = [SpeciesFactory, FamilyFactory, ReactionFactory]

    def __init__(self, syntax=Syntax(), factories=None) -> None:
        self.syntax = syntax
        self.position_pattern = re.compile(f'^( +)?(.*?)([{syntax.period_equivalent}{syntax.colon_equivalent}])?$')
        self.header_pattern = re.compile(f'^([a-zA-Z]+) ([a-zA-Z0-9_\-> \+{syntax.family_denoter}]+)$')
        self.family_pattern = re.compile(f'{syntax.family_denoter}([a-zA-Z]+)')
        if factories:
            self.factories = factories
        self.factory_lookup = {d.header: d for d in self.factories}

    def localize_properties_with_family_members(self, atom_properties, family_names, member_choices):
        # member choices == list of ordered pairs (family name, which member)
        new_dictionary = atom_properties.copy()
        for parameter_name, parameter in new_dictionary.items():
            new_dictionary[parameter_name] = parameter.localize_with_family_members(self.syntax, family_names, member_choices)
        return new_dictionary

    def add_atoms(self, existing_atoms, factory, atom_name, atom_properties):
        new_atoms = {}
        if self.syntax.family_denoter in atom_name:
            families = set(re.findall(self.family_pattern, atom_name))
            print("HERE", families)
            family_members = []
            for family_name in families:
                family = existing_atoms.get(family_name, None)
                if family is None or not(isinstance(family, Family)):
                    raise ModelSyntaxError(f"looked for family {family_name} but couldn't find its definition.")
                # append the *list*, so we can keep our families straight
                family_members.append(family.members)
            for combination in product(*family_members):
                print(combination)
                localized_name = PropertyMatch.localize_string_with_family_members(atom_name, self.syntax, families, combination)
                localized_properties = self.localize_properties_with_family_members(atom_properties, families, combination)
                new_atoms[localized_name] = self.construct_atom(existing_atoms, factory, localized_name, localized_properties)
        else:
            new_atoms[atom_name] = self.construct_atom(existing_atoms, factory, atom_name, atom_properties)
        existing_atoms.update(new_atoms)
        return existing_atoms

    def construct_atom(self, atoms, factory, atom_name, atom_properties):
        if atoms.get(atom_name, None) is not None:
            raise DuplicateAtomNameError(f"Duplicate atom name {atom_name}.")
        return factory.construct(atom_name, atom_properties, atoms)

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
        atom_factory = None
        atom_name = None
        whitespace = None
        for i,l in enumerate(lines):
            try:
                postion_match = re.match(self.position_pattern, l)
                self.check_match(postion_match, l, "a valid line")
                indent, line_body, terminator = postion_match[1], postion_match[2], postion_match[3]
                #print(i)
                #print(indent, line_body, terminator)

                # check if indent is acceptable
                if indent and expect_header:
                    raise ModelSyntaxError(f"Unexpected indent.")

                # if blank line, verify that is acceptable and then pipe all previous atom lines together
                if l==self.syntax.atom_separator:
                    if not expect_blank:
                        raise ModelSyntaxError(f"Unexpected blank line. Did you terminate this unit with a {self.syntax.period_equivalent}?")

                    # build (potentially several -- if family) new atoms from name, properties, and all the existing atoms
                    atoms = self.add_atoms(atoms, atom_factory, atom_name, atom_properties)

                    expect_blank = False
                    expect_header = True
                    atom_properties = {}
                    atom_header = None
                    atom_name = None
                    atom_factory = None
                    continue

                if expect_header:
                    match = re.match(self.header_pattern, line_body)
                    print(line_body)
                    self.check_match(match, line_body, 'ObjectType Name:')
                    atom_header = match[1]
                    atom_name = match[2]
                    try:
                        atom_factory = self.factory_lookup[atom_header]
                    except KeyError:
                        raise ModelSyntaxError(f"No means to create {atom_header} (no factory).")
                    expect_header = False
                    expect_blank = (terminator == self.syntax.period_equivalent)
                    continue

                if whitespace is None:
                    whitespace = indent
                if whitespace != indent:
                    raise ModelSyntaxError(f'Inconsistent whitespace before property: value pair: "{l}"')

                for property_parser in atom_factory.properties:
                    match = property_parser.parse(line_body, self.syntax)
                    if match is not None:
                        break

                self.check_match(match, line_body, f'keyword: value (for the predefined properties of {atom_header})')
                atom_properties[match.property_name] = match
                expect_blank = (terminator == self.syntax.period_equivalent)
            except Exception as e:
                print(f"While parsing L:{i+1}:")
                raise e
            i+=1

        # for loop over; we are out of lines
        if not expect_blank:
            raise ModelSyntaxError(f'ran out of lines while parsing a single unit. Every Species/Reaction/Model should have its last line terminated by a "{self.syntax.period_equivalent}".')

        # make our last atom!
        atoms = self.add_atoms(atoms, atom_factory, atom_name, atom_properties)

        return atoms

    @staticmethod
    def check_match(match, line, expected):
        if match is None:
            raise ModelSyntaxError(f'expected to find a {expected} but found "{line}".')

if __name__ == '__main__':
    import sys
    p = Parser()
    atoms = p.parse_file(sys.argv[1])
    for name, atom in atoms.items():
        print("NAME:", name)
        print("TYPE:", type(atom))
        print(atom)