from dataclasses import dataclass
import itertools
from enum import Enum

class ReactionRateFamilyApplicationMethod(Enum):
    group = 'group'
    split = 'split'

def enumerated_product(members):
    yield from zip(itertools.product(*(range(len(x)) for x in members)), itertools.product(*members))


class FamilyConstraint():
    def __init__(self, families):
        self.families = families

    @staticmethod
    def find_relevant_entries(combination, family_labels, desired_names):
        return [combination[family_labels.index(n)] for n in desired_names]

    def check_valid(self, family_labels, combination):
        # Returns True if combination is acceptable under constraint
        for s in self.families:
            entries = self.find_relevant_entries(combination, family_labels, s)
            for x,y in itertools.combinations(entries, 2):
                if not self.valid(x, y):
                    return False
        return True

class EqualsConstraint(FamilyConstraint):
    def valid(self, x, y):
        return x == y

class NotEqualsConstraint(FamilyConstraint):
    def valid(self, x, y):
        return x != y

constraint_by_name = {
    '!=': NotEqualsConstraint,
    '==': EqualsConstraint,
}

@dataclass(frozen=True)
class Syntax():
    family_denoter = '$'
    family_enumerator = '#'
    family_alternator = '@'

    def family_replace(self, family_names, idx, chosen_members, value, do_nestings=False, paired_family_mapping=None):
        if paired_family_mapping is None:
            paired_family_mapping = {}
        for family_name, member, i in zip(family_names, chosen_members, idx):
            if do_nestings and isinstance(value, (tuple, list)):
                value = [self.family_replace(family_names, idx, chosen_members, v, do_nestings, paired_family_mapping=paired_family_mapping) for v in value]
                continue
            if do_nestings and isinstance(value, dict):
                value = {k:self.family_replace(family_names, idx, chosen_members, v, do_nestings, paired_family_mapping=paired_family_mapping) for k,v in value.items()}
                continue
            if isinstance(value, (int, float)):
                continue
            if paired_family_mapping.get(family_name):
                denoted_member = member[0]
                alternated_member = member[1]
                value = value.replace(self.family_alternator + family_name, alternated_member)
            else:
                denoted_member = member
                if self.family_alternator in value:
                    print(f"WARNING: found family alternator symbol {self.family_alternator} but no family was paired.")
            value = value.replace(self.family_denoter + family_name, denoted_member)
            value = value.replace(self.family_enumerator + family_name, str(i))

        return value

    def expand_families(self, families, atom):
        if not isinstance(atom, dict):
            raise TypeError(f"Instead of dictionary found {type(atom)}. Did you remember to prepend '-' to list items in yaml?")
        try:
            used_families = atom.pop('used_families')
        except KeyError:
            return [atom]

        constraints = []
        family_constraints = atom.pop('family_constraints', None)
        if family_constraints is not None:
            if isinstance(family_constraints, dict):
                family_constraints = [family_constraints]
            for c_data in family_constraints:
                constraint_name = c_data.pop('constraint')
                constraint_klass = constraint_by_name[constraint_name]
                constraint = constraint_klass(**c_data)
                constraints.append(constraint)

        nested_fields = ['products', 'reactants']
        # are we a ReactionRateFamily?
        if 'reactions' in atom.keys():
            # we are a ReactionRateFamily
            try:
                family_method = ReactionRateFamilyApplicationMethod(atom.pop('family_method'))
            except KeyError:
                family_method = ReactionRateFamilyApplicationMethod.group

            if family_method == ReactionRateFamilyApplicationMethod.group:
                for field, value in atom.items():
                    if field != 'reactions':
                        assert self.family_denoter not in value
                        assert self.family_enumerator not in value
                        continue

                new_reactions = []
                for r in atom['reactions']:
                    r_ = r.copy()
                    r_['used_families'] = used_families
                    new = self.expand_families(families, r_, self)
                    new_reactions.extend(new)
                new_rrf = atom.copy()
                new_rrf['reactions'] = new_reactions
                return new_rrf
            elif family_method == ReactionRateFamilyApplicationMethod.split:
                # we want to continue with expansion, but make a note that reactions will have to be expanded as well
                nested_fields.append('reactions')
        family_members = []
        paired_family_mapping = {}
        for fam, global_family_name in used_families.items():
            # we might have a paired family instead of a single family
            if isinstance(global_family_name, list):
                assert len(global_family_name) == 2, f"Only specifying exactly 2 paired families is supported. Got {global_family_name}"
                members = tuple(zip(families[global_family_name[0]], families[global_family_name[1]]))
                paired_family_mapping[fam] = True
            else:
                members = families[global_family_name]
            # append the *list*, so we can keep our families straight
            family_members.append(members)

        new_atoms = []
        for idx, combination in enumerated_product(family_members):
            for c in constraints:
                if not c.check_valid(list(used_families.keys()), combination):
                    continue
            new_atom = {}
            for field, value in atom.items():
                nested = field in nested_fields
                if nested: assert isinstance(value, list), f"For nested field {field} found flat data. Did you remember to include [] in specifying a list of length 1?"
                new_field = self.family_replace(used_families, idx, combination, field, do_nestings=False, paired_family_mapping=paired_family_mapping)
                new_atom[new_field] = self.family_replace(used_families, idx, combination, value, do_nestings=(field in nested_fields), paired_family_mapping=paired_family_mapping)
            new_atoms.append(new_atom)

        return new_atoms