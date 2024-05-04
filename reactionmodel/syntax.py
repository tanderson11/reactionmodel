from dataclasses import dataclass
from itertools import product
from enum import Enum

class ReactionRateFamilyApplicationMethod(Enum):
    group = 'group'
    split = 'split'

def enumerated_product(members):
    yield from zip(product(*(range(len(x)) for x in members)), product(*members))

@dataclass(frozen=True)
class Syntax():
    family_denoter = '$'
    family_enumerator = '#'

    def family_replace(self, family_names, idx, chosen_members, value, do_nestings=False):
        for family_name, member, i in zip(family_names, chosen_members, idx):
            if do_nestings and isinstance(value, (tuple, list)):
                value = [self.family_replace(family_names, idx, chosen_members, v, do_nestings) for v in value]
                continue
            if do_nestings and isinstance(value, dict):
                value = {k:self.family_replace(family_names, idx, chosen_members, v, do_nestings) for k,v in value.items()}
                continue
            if isinstance(value, (int, float)):
                continue
            value = value.replace(self.family_denoter + family_name, member)
            value = value.replace(self.family_enumerator + family_name, str(i))
        return value

    def expand_families(self, families, atom):
        try:
            used_families = atom.pop('used_families')
        except KeyError:
            return [atom]
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
        for _, global_family_name in used_families.items():
            members = families[global_family_name]
            # append the *list*, so we can keep our families straight
            family_members.append(members)

        new_atoms = []
        for idx, combination in enumerated_product(family_members):
            new_atom = {}
            for field, value in atom.items():
                nested = field in nested_fields
                if nested: assert isinstance(value, list), f"For nested field {field} found flat data. Did you remember to include [] in specifying a list of length 1?"
                new_field = self.family_replace(used_families, idx, combination, field, do_nestings=False)
                new_atom[new_field] = self.family_replace(used_families, idx, combination, value, do_nestings=(field in nested_fields))
            new_atoms.append(new_atom)

        return new_atoms