import json
import numpy as np

with open('data/report6/param/network/perturbated3.json', 'r', encoding='utf-8') as file:
    perturbated = json.load(file)

with open('data/report6/param/network/original.json', 'r', encoding='utf-8') as file:
    original = json.load(file)

original['frequency'] = perturbated['frequency']

o_connectivity = np.array(original['connectivity'])
p_connectivity = np.array(perturbated['connectivity'])
a11 = o_connectivity[0:10, 0:10]
a22 = o_connectivity[10:20, 10:20]
a33 = o_connectivity[20:30, 20:30]

a12 = p_connectivity[0:10, 10:20]
a21 = a12.T
a23 = p_connectivity[10:20, 20:30]
a32 = a23.T
a13 = o_connectivity[0:10, 20:30]
a31 = a13.T
original['connectivity'] = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]).tolist()

with open('data/report6/param/network/original_perturbated.json', 'w', encoding='utf-8') as file:
    json.dump(original, file, indent=4)