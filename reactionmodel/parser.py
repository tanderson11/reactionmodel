import re
import os
import ast
from itertools import product
from dataclasses import dataclass

class ParserSyntaxError(Exception):
    pass

class DuplicateAtomNameError(Exception):
    pass

class MissingRequiredPropertyError(Exception):
    pass

class UnexpectedPropertyError(Exception):
    pass

@dataclass(frozen=True)
class Syntax():
    atom_separator = '\n'
    list_delimiter = ','
    colon_equivalent = ':'
    period_equivalent = '.'
    family_denoter = '$'
    reaction_arrow = '->'
    family_enumerator = '#'
    null_set = 'NULL'

class PropertyMatch():
    def __init__(self, property_name, value, used_in_constructor):
        self.property_name = property_name
        self.value = value
        self.used_in_constructor = used_in_constructor

    @staticmethod
    def localize_string_with_family_members(string, syntax, families, chosen_members, idx):
        for family_name, member, i in zip(families, chosen_members, idx):
            string = string.replace(syntax.family_denoter + family_name, member)
            string = string.replace(syntax.family_enumerator + family_name, str(i))
        return string

    def localize_value_with_family_members(self, syntax, families, chosen_members, idx):
        return self.localize_string_with_family_members(self.value, syntax, families, chosen_members, idx)

    def localize_with_family_members(self, syntax, families, chosen_members, idx):
        value = self.localize_value_with_family_members(syntax, families, chosen_members, idx)
        return self.__class__(self.property_name, value, self.used_in_constructor)

    def evaluate_with_existing_atoms(self, existing_atoms):
        return self.value

class ExpressionMatch(PropertyMatch):
    pass

class DictionaryMatch(PropertyMatch):
    def localize_value_with_family_members(self, syntax, families, chosen_members, idx):
        new_value = {
            self.localize_string_with_family_members(k, syntax, families, chosen_members, idx):self.localize_string_with_family_members(v, syntax, families, chosen_members, idx)
            for k,v in self.value.items()
        }
        return new_value

class UsedFamiliesMatch(DictionaryMatch):
    def evaluate_with_existing_atoms(self, existing_atoms):
        #import pdb; pdb.set_trace()
        evaluated = {k:existing_atoms[v] for k,v in self.value.items()}
        return evaluated

class ListMatch(PropertyMatch):
    def localize_value_with_family_members(self, syntax, families, chosen_members, idx):
        new_value = [self.localize_string_with_family_members(v, syntax, families, chosen_members, idx) for v in self.value]
        return new_value

class Property():
    value_pattern_string = '([a-zA-Z0-9\.]+)$'
    match_klass = PropertyMatch
    used_in_constructor = True
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
            return self.match_klass(match[1], match[2], self.used_in_constructor)
        return None

class RichProperty(Property):
    value_pattern_string = '"(.*)"$'

class PathProperty(Property):
    value_pattern_string = '"(.*)"$'

class ExpressionProperty(Property):
    value_pattern_string = '(.*)$'
    match_klass = ExpressionMatch

class DictionaryProperty(Property):
    value_pattern_string = '(.*)$'
    match_klass = DictionaryMatch

    def parse(self, line, syntax):
        property_match = super().parse(line, syntax)
        # parse the dictionary
        if property_match is not None:
            property_match.value = ast.literal_eval(property_match.value)
        return property_match

class UsedFamiliesProperty(DictionaryProperty):
    used_in_constructor = False
    match_klass = UsedFamiliesMatch

class ListProperty(Property):
    value_pattern_string = '([a-zA-Z0-9<>\(\)_ {0}]+)$'
    match_klass = ListMatch

    def inject_syntax(self, syntax):
        return self.value_pattern_string.format(re.escape(syntax.list_delimiter))

    def parse(self, line, syntax):
        property_match = super().parse(line, syntax)
        # split the list!
        if property_match is not None:
            value = property_match.value.strip(' ')
            property_match.value = value.split(syntax.list_delimiter)
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
            if v.used_in_constructor:
                ps.append(v.evaluate_with_existing_atoms(existing_atoms))
        for p,v in property_matches.items():
            if p not in optional_properties:
                raise UnexpectedPropertyError(f'{name} had unexpected property {p}.')
            if v.used_in_constructor:
                optional_ps[p] = v.evaluate_with_existing_atoms(existing_atoms)

        return cls.from_properties(name, *ps, **optional_ps)

    @classmethod
    def from_properties(cls, name, *properties, **optional_properties):
        return cls.klass(name, *properties, **optional_properties)

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

class Parser():
    def __init__(self, syntax=Syntax()) -> None:
        self.syntax = syntax
        self.position_pattern = re.compile(f'^( +)?(.*?)([{re.escape(syntax.period_equivalent)}{re.escape(syntax.colon_equivalent)}])?$')
        self.header_pattern = re.compile(f'^([a-zA-Z]+) ([a-zA-Z0-9_\-><\(\) \+{re.escape(syntax.family_denoter)}]+)$')
        self.family_pattern = re.compile(f'{re.escape(syntax.family_denoter)}([a-zA-Z]+)')

    def localize_properties_with_family_members(self, atom_properties, family_names, member_choices, idx):
        # member choices == list of ordered pairs (family name, which member)
        new_dictionary = atom_properties.copy()
        for property_name, property in new_dictionary.items():
            new_dictionary[property_name] = property.localize_with_family_members(self.syntax, family_names, member_choices, idx)
        return new_dictionary

    @staticmethod
    def enumerated_product(*args):
        yield from zip(product(*(range(len(x)) for x in args)), product(*args))

    def add_atoms(self, existing_atoms, factory, atom_name, atom_properties):
        new_atoms = {}
        if self.syntax.family_denoter in atom_name:
            #import pdb; pdb.set_trace()
            families = set(re.findall(self.family_pattern, atom_name))
            families = re.findall(self.family_pattern, atom_name)

            used_families_property_name = None
            for p in factory.properties:
                if isinstance(p, UsedFamiliesProperty):
                    used_families_property_name = p.name
            if used_families_property_name is None:
                raise ParserSyntaxError(f"atom {atom_name} used the family token {self.syntax.family_denoter} but that atom type doesn't support families.")

            used_families_property = atom_properties.get(used_families_property_name, None)
            if used_families_property is None:
                raise ParserSyntaxError(f"atom {atom_name} used the family token {self.syntax.family_denoter} but no {used_families_property_name} property was specified.")

            # teach the used families property about the families that exist in our specification
            used_families_property = used_families_property.evaluate_with_existing_atoms(existing_atoms)

            family_members = []
            for f in families:
                family = used_families_property.get(f, None)
                if family is None or not(isinstance(family, Family)):
                    raise ParserSyntaxError(f"looked for family {f} but couldn't find its alias in {used_families_property_name}.")
                # append the *list*, so we can keep our families straight
                family_members.append(family.members)
            for idx, combination in self.enumerated_product(*family_members):
                localized_name = PropertyMatch.localize_string_with_family_members(atom_name, self.syntax, families, combination, idx)
                localized_properties = self.localize_properties_with_family_members(atom_properties, families, combination, idx)
                new_atoms[localized_name] = self.construct_atom(existing_atoms, factory, localized_name, localized_properties)
        else:
            new_atoms[atom_name] = self.construct_atom(existing_atoms, factory, atom_name, atom_properties)
        #import pdb; pdb.set_trace()
        existing_atoms.update(new_atoms)
        return existing_atoms

    def construct_atom(self, atoms, factory, atom_name, atom_properties):
        if atoms.get(atom_name, None) is not None:
            raise DuplicateAtomNameError(f"Duplicate atom name {atom_name}.")
        return factory.construct(atom_name, atom_properties, atoms)

    def parse_file(self, file, available_factories):
        old_dir = os.getcwd()
        path = os.path.abspath(file)
        os.chdir(os.path.dirname(path))
        with open(path, 'r') as f:
            raw = f.readlines()
        parsed = self.parse_lines(raw, available_factories)
        os.chdir(old_dir)
        return parsed

    def parse_lines(self, lines, available_factories):
        factory_lookup = {f.header: f for f in available_factories}
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

                # check if indent is acceptable
                if indent and expect_header:
                    raise ParserSyntaxError(f"Unexpected indent.")
                
                if expect_header and terminator != self.syntax.colon_equivalent:
                    raise ParserSyntaxError(f"Introduction of new item didn't end in colon or period.")

                # if blank line, verify that is acceptable and then pipe all previous atom lines together
                if l==self.syntax.atom_separator:
                    if not expect_blank:
                        raise ParserSyntaxError(f"Unexpected blank line. Did you terminate this unit with a {self.syntax.period_equivalent}?")

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
                    self.check_match(match, line_body, 'ObjectType Name:')
                    atom_header = match[1]
                    atom_name = match[2]
                    try:
                        atom_factory = factory_lookup[atom_header]
                    except KeyError:
                        raise ParserSyntaxError(f"No means to create {atom_header} (no factory).")
                    expect_header = False
                    expect_blank = (terminator == self.syntax.period_equivalent)
                    continue

                if whitespace is None:
                    whitespace = indent
                if whitespace != indent:
                    raise ParserSyntaxError(f'Inconsistent whitespace before property: value pair: "{l}"')

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
            raise ParserSyntaxError(f'ran out of lines while parsing a single unit. Every Species/Reaction/Model should have its last line terminated by a "{self.syntax.period_equivalent}".')

        # make our last atom!
        atoms = self.add_atoms(atoms, atom_factory, atom_name, atom_properties)

        return atoms

    @staticmethod
    def check_match(match, line, expected):
        if match is None:
            raise ParserSyntaxError(f'expected to find a {expected} but found "{line}".')