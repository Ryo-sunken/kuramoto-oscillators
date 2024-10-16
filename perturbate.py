import sys
import os
import json
import random
import numpy as np

CLUSTER_NODES = [10, 10, 10]
FREQUENCY = [5.0, 10.0, 15.0]
FREQUENCY_SIGMA = 1.0
STATE_DIM = 30
INPUT_DIM = 30

def main():
    with open('data/report6/param/network/damaged.json', 'r', encoding='utf-8') as file:
        param = json.load(file)
    
    connectivity = np.array(param['connectivity'])

    a11 = connectivity[0:10, 0:10]
    a22 = connectivity[10:20, 10:20]
    a33 = connectivity[20:30, 20:30]

    a12 = weighted_inter(0, 1, 0.1, 0.1)
    a13 = np.zeros((CLUSTER_NODES[0], CLUSTER_NODES[2]))
    a21 = a12.T
    a23 = weighted_inter(1, 2, 0.1, 0.1)
    a31 = a13.T
    a32 = a23.T

    param['connectivity'] = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]).tolist()

    param['frequency'] = create_frequency()

    with open('data/report6/param/network/perturbated.json', 'w', encoding='utf-8') as file:
        json.dump(param, file, indent=4)

def weighted_inter(cluster1, cluster2, min, r):
    adj = np.zeros((CLUSTER_NODES[cluster1], CLUSTER_NODES[cluster2]))
    for i in range(CLUSTER_NODES[cluster1]):
        for j in range(CLUSTER_NODES[cluster2]):
            if np.random.random() >= 0.5:
                adj[i, j] = np.random.random() * r + min
    return adj

def create_frequency():
    freq = []
    for k in range(len(CLUSTER_NODES)):
        freq_k = np.random.normal(FREQUENCY[k], FREQUENCY_SIGMA, CLUSTER_NODES[k])
        freq.extend(freq_k)
    return freq


if __name__ == '__main__':
    main()