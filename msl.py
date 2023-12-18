import re
from reactionmodel import Species, Reaction, Model
from dataclasses import dataclass

# model specification language

class AtomDecoder():
    klass = None
    properties = []
    optional_properties = []

    @classmethod
    def decode(cls, name, properties, existing_atoms={}):
        ps = []
        optional_ps = {}

        # go through required properties IN ORDER
        for p in cls.properties:
            try:
                v = properties.pop(p)
            except KeyError:
                raise MissingRequiredPropertyError(f'{name} is missing {p}')
            ps.append(v)
        for p,v in properties.items():
            if p not in cls.optional_properties:
                raise UnexpectedPropertyError(f'{name} had unexpected property {p}')
            optional_ps[p] = v
        
        return cls.klass(name, *ps, **optional_ps)


class FamilyDecoder(AtomDecoder):
    header = "Family"
    properties = ['members']
    optional_properties = ['description']

class SpeciesDecoder(AtomDecoder):
    klass = Species
    header = "Species"
    properties = []
    optional_properties = ['description']

class ReactionDecoder(AtomDecoder):
    klass = Reaction
    header = "Reaction"
    properties = ['description', 'reactants', 'products']
    optional_properties = ['rate_involvement', 'k', 'reversible']

class ModelDecoder(AtomDecoder):
    klass = Model
    header = "Model"

class ModelSyntaxError(Exception):
    pass

class MissingRequiredPropertyError(Exception):
    pass

class UnexpectedPropertyError(Exception):
    pass

@dataclass(frozen=True)
class Syntax():
    atom_separator = '\n'
    family_denoter = '_'
    list_delimiter = ','
    colon_equivalent = ':'
    period_equivalent = '.'

class Parser():
    decoders = [SpeciesDecoder]

    def __init__(self, syntax=Syntax(), decoders=None) -> None:
        self.syntax = syntax
        self.header_pattern = re.compile(f'^([a-zA-Z]+) ([a-zA-Z0-9_]+)$')
        self.line_pattern = re.compile(f'^([a-z]+): ([a-zA-Z0-9]+)$')
        self.description_pattern = re.compile(f'^(description): "(.*)"$')
        self.position_pattern = re.compile(f'^( +)?(.*)([{syntax.period_equivalent}{syntax.colon_equivalent}])')
        if decoders:
            self.decoders = decoders
        self.decoder_lookup = {d.header: d for d in self.decoders}

    def make_atom(self, atom_header, atom_name, atom_dictionary, atoms, i):
        try:
            relevant_decoder = self.decoder_lookup[atom_header]
        except KeyError:
            raise ModelSyntaxError(f"No decoder found for {atom_header}. L:{i+1}")
        
        return relevant_decoder.decode(atom_name, atom_dictionary, atoms)

    def parse_file(self, file):
        with open(file, 'r') as f:
            raw = f.readlines()
        return self.parse_lines(raw)
    
    def parse_lines(self, lines):
        expect_header = True
        expect_blank = False
        
        i = 0
        atoms = []
        atom_dictionary = {}
        atom_header = None
        atom_name = None
        whitespace = None
        for i,l in enumerate(lines):
            #print(l, l=='', l=='\n', l=='\r')

            line_match = re.match(self.position_pattern, l)
            self.check_match(line_match, l, i+1, "a valid line")
            indent, line_body, terminator = line_match[1], line_match[2], line_match[3]

            # check if indent is acceptable
            if indent and expect_header:
                raise ModelSyntaxError(f"Unexpected indent. L:{i+1}")

            # if blank line, verify that is acceptable and then pipe all previous atom lines together
            if l==self.syntax.atom_separator:
                if not expect_blank:
                    raise ModelSyntaxError(f"Unexpected blank line. L:{i+1}")
                print(atom_header)

                # build a new atom from name, properties, and all the existing atoms
                atom = self.make_atom(atom_header, atom_name, atom_dictionary, atoms, i)
                atoms.append(atom)

                expect_blank = False
                expect_header = True
                atom_dictionary = {}
                atom_header = None
                atom_name = None
                continue

            if expect_header:
                match = re.match(self.header_pattern, line_body)
                self.check_match(match, line_body, i+1, 'Object name(:optional):')
                atom_header = match[1]
                atom_name = match[2]
                expect_header = False
                expect_blank = (terminator == self.syntax.period_equivalent)
                continue

            # description lines are special: they should be in quotes and can contain arbitrary characters
            match = re.match(self.description_pattern, line_body)
            # but if we don't find a description, try our generic "property: value" line pattern
            if match is None:
                match = re.match(self.line_pattern, line_body)
                self.check_match(match, line_body, i+1, 'keyword: value')
            if whitespace is None:
                whitespace = indent
            if whitespace != indent:
                raise ModelSyntaxError(f'Inconsistent whitespace before property: value pair: "{l}" on L{i+1}')
            atom_dictionary[match[1]] = match[2]
            expect_blank = (terminator == self.syntax.period_equivalent)
            i+=1
        if not expect_blank:
            raise ModelSyntaxError(f'ran out of lines while parsing a single unit. Every Species/Reaction/Model should have its last line terminated by a {self.syntax.period_equivalent}')
        
        # make our last atom!
        atom = self.make_atom(atom_header, atom_name, atom_dictionary, atoms, i)
        atoms.append(atom)

        print(atoms)
    
    @staticmethod
    def check_match(match, line, i, expected):
        if match is None:
            raise ModelSyntaxError(f'expected to find a {expected} but found "{line}" on L:{i}')

if __name__ == '__main__':
    import sys
    p = Parser()
    p.parse_file(sys.argv[1])