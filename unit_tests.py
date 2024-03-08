import unittest
from itertools import product
import os
import numpy as np

from reactionmodel.model import Species, Model, Reaction
import reactionmodel.parser as parser

class LoadModelTestCase(unittest.TestCase):
    def load_from_file(self, path):
        return parser.load(os.path.join(path, 'model.yaml')).model

    def test_minimal(self):
        path = './examples/minimal/'
        A = Species('A', description="a lengthy description of A")
        B = Species('B')
        C = Species('C')

        r1 = Reaction([A, B], [C], k=2.0, description="A + B => C (rate constant 2.0)")
        r2 = Reaction([C], [], k=0.5, description="C => empty set (rate constant 0.5)")

        m = Model([A,B,C], [r1,r2])

        self.assertEqual(m, self.load_from_file(path))
    
    def test_family(self):
        path = './examples/family_and_matrix/'

        As = ['x', 'y', 'z']
        Bs = ['p', 'q', 'r']

        Q = np.array(range(9)).reshape(3,3)

        A_names = [f'A_{a}' for a in As]
        B_names = [f'B_{b}' for b in Bs]
        a_species   = [Species(a) for a in A_names]
        b_species   = [Species(b) for b in B_names]

        reactions = [
            Reaction(
                [(a, 2)],
                b,
                description=f'2{a.name} => {b.name}',
                k=f'Q[{A_names.index(a.name)}][{B_names.index(b.name)}]'
            )
            for a,b in product(a_species, b_species)
        ]
        m = Model(a_species+b_species, reactions)

        self.assertEqual(m, self.load_from_file(path))

    def test_double_family(self):
        strains = ['X', 'Y']
        species = []
        for i,j,k in product(strains, strains, strains):
            species.append(Species(f'<{i}_{j}_{k}>'))

        reactions = []
        for s in species:
            reactions.append(Reaction(reactants=[s], products=[], description=f"death of {s.name}"))

        m = Model(species, reactions)

        path = './examples/double_used_family'
        self.assertEqual(m, self.load_from_file(path))
