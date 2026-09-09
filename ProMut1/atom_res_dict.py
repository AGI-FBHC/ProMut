# 氨基酸字典
res_atom_type_dict = {
    'GLY': ['N', 'CA', 'C', 'O'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
}

label_atom_type_dict = {
    18: {'N', 'CA', 'C', 'O', 'H', 'HA2', 'HA3'},
    19: {'N', 'CA', 'C', 'O', 'CB', 'SG', 'H', 'HA', 'HB2', 'HB3'},
    2: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2',
        'HD3', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'},
    5: {'N', 'CA', 'C', 'O', 'CB', 'OG', 'H', 'HA', 'HB2', 'HB3', 'HG'},
    6: {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'H', 'HA', 'HB', 'HG21', 'HG22', 'HG23', 'HG1'},
    1: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2',
        'HE3', 'HZ1', 'HZ2', 'HZ3'},
    13: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'},
    9: {'N', 'CA', 'C', 'O', 'CB', 'H', 'HA', 'HB1', 'HB2', 'HB3'},
    11: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'H', 'HA', 'HB2', 'HB3', 'HG', 'HD11', 'HD12', 'HD13', 'HD21',
         'HD22', 'HD23'},
    12: {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'H', 'HA', 'HB', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23',
         'HD11', 'HD12', 'HD13'},
    10: {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'H', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'},
    3: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'H', 'HA', 'HB2', 'HB3'},
    4: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3'},
    0: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'H', 'HA', 'HB2', 'HB3', 'ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HD2', 'HE1'},
    7: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'H', 'HA', 'HB2', 'HB3', 'OD1', 'ND2', 'HD22', 'HD21'},
    17: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'},
    8: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'H', 'HA', 'HB2', 'HB3', 'HG2', 'HG3', 'OE1', 'NE2', 'HE21', 'HE22'},
    14: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2',
         'HE1', 'HE2', 'HZ'},
    16: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'H', 'HA', 'HB2',
         'HB3', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'},
    15: {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2',
         'HE1', 'HE2', 'HH'},
}

res_group_dict = {
    0: 'group1', 1: 'group1', 2: 'group1',
    3: 'group2', 4: 'group2',
    5: 'group3', 6: 'group3', 7: 'group3', 8: 'group3',
    9: 'group4', 10: 'group4', 11: 'group4', 12: 'group4', 13: 'group4',
    14: 'group5', 15: 'group5', 16: 'group5',
    17: 'group6', 18: 'group6',
    19: 'group7'
}


abrev = {'HIS': 'H', 'LYS': 'K', 'ARG': 'R', 'ASP': 'D', 'GLU': 'E', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q',
         'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'MET': 'M', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W', 'PRO': 'P',
         'GLY': 'G', 'CYS': 'C'}

label_res_dict = {0: 'HIS', 1: 'LYS', 2: 'ARG', 3: 'ASP', 4: 'GLU', 5: 'SER', 6: 'THR', 7: 'ASN', 8: 'GLN', 9: 'ALA',
                  10: 'VAL', 11: 'LEU', 12: 'ILE', 13: 'MET', 14: 'PHE', 15: 'TYR', 16: 'TRP', 17: 'PRO', 18: 'GLY',
                  19: 'CYS'}
res_label_dict = {'HIS': 0, 'LYS': 1, 'ARG': 2, 'ASP': 3, 'GLU': 4, 'SER': 5, 'THR': 6, 'ASN': 7, 'GLN': 8, 'ALA': 9,
                  'VAL': 10, 'LEU': 11, 'ILE': 12, 'MET': 13, 'PHE': 14, 'TYR': 15, 'TRP': 16, 'PRO': 17, 'GLY': 18,
                  'CYS': 19}

letter1_3_dict = {'H': 'HIS', 'K': 'LYS', 'R': 'ARG', 'D': 'ASP', 'E': 'GLU', 'S': 'SER', 'T': 'THR', 'N': 'ASN',
                  'Q': 'GLN', 'A': 'ALA', 'V': 'VAL', 'L': 'LEU', 'I': 'ILE', 'M': 'MET', 'F': 'PHE', 'Y': 'TYR',
                  'W': 'TRP', 'P': 'PRO', 'G': 'GLY', 'C': 'CYS'}

bias = [-0.558, -0.73, 1.226]
