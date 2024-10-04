import numpy as np
import matplotlib.pyplot as plt
import json
import sys

args = sys.argv

path = 'data/' + args[1] + '/param/network/' + args[2] + '.json'
with open(path, mode='r', encoding='utf-8') as file:
    param = json.load(file)

path = 'data/' + args[1] + '/param/control/' + args[3] + '.json'
with open(path, mode='r', encoding='utf-8') as file:
    control = json.load(file)


frequency = np.array(param['frequency'])
connectivity = np.array(param['connectivity'])
gain = np.array(control['input_weight'])

del_freq = np.array([0.0, 0.0, 0.0])
a_max = np.array([0.0, 0.0, 0.0])
a_min = np.array([0.0, 0.0, 0.0])
d_min = np.array([0.0, 0.0, 0.0])
D_min = np.array([0.0, 0.0, 0.0])
D_max = np.array([0.0, 0.0, 0.0])
g_max = np.array([0.0, 0.0, 0.0])
g_min = np.array([0.0, 0.0, 0.0])
epsilon = np.array([0.0, 0.0, 0.0])
f_intra_min = np.array([0.0, 0.0, 0.0])

node_sums = [0, 10, 20, 30]

def f_intra_k_max(psi, k):
    return - a_max[k] * 10 * np.sin(psi) + 2 * (a_max[k] * 9 - 2 * D_min[k])

def f_intra_k_min(psi, k):
    return - a_min[k] * d_min[k] * np.sin(psi)

def f_intra_k(psi, k):
    return np.minimum(f_intra_k_max(psi, k), f_intra_k_min(psi, k))

def f_inter_k_EEP(psi, k):
    return 2 * D_max[k] * psi + epsilon[k]

def f_inter_k(psi, k):
    return np.minimum(2 * D_max[k], f_inter_k_EEP(psi, k))

def f_input_k(psi, k):
    return -g_max[k] * np.sin(psi) + 2 * (g_max[k] - g_min[k])

for k in range(0, 3):
    intra_connect = connectivity[node_sums[k]:node_sums[k+1], node_sums[k]:node_sums[k+1]]
    if k == 0:
        inter_connect1 = connectivity[node_sums[k]:node_sums[k+1], 10:20]
        inter_connect2 = connectivity[node_sums[k]:node_sums[k+1], 20:30]
    if k == 1:
        inter_connect1 = connectivity[node_sums[k]:node_sums[k+1], 0:10]
        inter_connect2 = connectivity[node_sums[k]:node_sums[k+1], 20:30]
    if k == 2:
        inter_connect1 = connectivity[node_sums[k]:node_sums[k+1], 0:10]
        inter_connect2 = connectivity[node_sums[k]:node_sums[k+1], 10:20]

    intra_gain = gain[node_sums[k]:node_sums[k+1]]

    e_max1 = inter_connect1.sum(axis=0).max()
    e_max2 = inter_connect2.sum(axis=0).max()
    e_min1 = inter_connect1.sum(axis=0).min()
    e_min2 = inter_connect2.sum(axis=0).min()

    intra_freq = frequency[node_sums[k]:node_sums[k+1]]
    del_freq[k] = intra_freq.max() - intra_freq.min()
    a_max[k] = intra_connect.max()
    a_min[k] = intra_connect[intra_connect > 0.0].min()
    d_min[k] = (intra_connect > 0.0).sum(axis=0).min()
    D_max[k] = (inter_connect1.sum(axis=0) + inter_connect2.sum(axis=0)).max()
    D_min[k] = intra_connect.sum(axis=0).min()
    g_max[k] = intra_gain.max()
    g_min[k] = intra_gain.min()
    epsilon[k] = e_max1 - e_min1 + e_max2 - e_min2
    f_intra_min[k] = min(a_max[k] * 8 - 2 * D_min[k], - a_min[k] * d_min[k])
    
psi = np.arange(0, np.pi, 0.01)

plt.plot(psi, -(f_intra_k(psi, int(args[4])) + f_input_k(psi, int(args[4]))))
plt.plot(psi, f_inter_k(psi, int(args[4])))

plt.show()