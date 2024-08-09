import sys
import os
import json
import random
import numpy as np
from my_functions import weighted_watts_strogatz_adjacency, weighted_complete_adjacency

CLUSTER_NODES = [10, 10, 10]
FREQUENCY = [5.0, 10.0, 15.0]
FREQUENCY_SIGMA = 1.0
STATE_DIM = 30
INPUT_DIM = 30


def main():
    args = sys.argv
    folder_name = args[1]

    # フォルダーの作成
    os.makedirs('data/' + folder_name + '/param/control', exist_ok=True)
    os.makedirs('data/' + folder_name + '/param/network', exist_ok=True)
    os.makedirs('data/' + folder_name + '/result', exist_ok=True)

    # パラメータファイルの作成
    network_param_path = 'data/' + folder_name + '/param/network/'
    comment = 'model:comp freq:[5 10 15] freq_s: 1.0 intra:1.0-2.0 inter:0.1-0.5 damage:random (0.5)'
    frequency = create_frequency()
    connectivity, _ = create_connectivity()

    param = dict()
    param['comment'] = comment
    param['state_dim'] = STATE_DIM
    param['input_dim'] = INPUT_DIM
    param['random_range'] = 1.0
    param['cluster_nodes_num'] = CLUSTER_NODES
    param['frequency'] = frequency
    param['connectivity'] = connectivity.tolist()

    with open(network_param_path + 'original.json', 'w', encoding='utf-8') as original:
        json.dump(param, original, indent=4)

    param['connectivity'] = damage(connectivity).tolist()

    with open(network_param_path + 'damaged.json', 'w', encoding='utf-8') as damaged:
        json.dump(param, damaged, indent=4)

def create_frequency():
    freq = []
    for k in range(len(CLUSTER_NODES)):
        freq_k = np.random.normal(FREQUENCY[k], FREQUENCY_SIGMA, CLUSTER_NODES[k])
        freq.extend(freq_k)
    return freq

def create_connectivity():
    # クラスター内結合の作成
    a11 = weighted_intra_model(0)
    a22 = weighted_intra_model(1)
    a33 = weighted_intra_model(2)

    # クラスター間結合の作成
    a12 = weighted_inter(0, 1, 0.1, 0.1)
    a13 = np.zeros((CLUSTER_NODES[0], CLUSTER_NODES[2]))
    a21 = a12.T
    a23 = weighted_inter(1, 2, 0.1, 0.1)
    a31 = a13.T
    a32 = a23.T

    ret1 = np.block([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    a12_2 = np.zeros((CLUSTER_NODES[0], CLUSTER_NODES[1]))
    a21_2 = a12_2.T
    a23_2 = np.zeros((CLUSTER_NODES[1], CLUSTER_NODES[2]))
    a32_2 = a23_2.T

    ret2 = np.block([[a11, a12_2, a13], [a21_2, a22, a23_2], [a31, a32_2, a33]])

    return ret1, ret2


def weighted_intra_model(cluster):
    return weighted_watts_strogatz_adjacency(CLUSTER_NODES[cluster], 7, 0.5, 1.0, 2.0)

def weighted_inter(cluster1, cluster2, min, r):
    adj = np.zeros((CLUSTER_NODES[cluster1], CLUSTER_NODES[cluster2]))
    for i in range(CLUSTER_NODES[cluster1]):
        for j in range(CLUSTER_NODES[cluster2]):
            if np.random.random() >= 0.5:
                adj[i, j] = np.random.random() * r + min
    return adj

def damage(connectivity):
    idxes = [(i, j) for i in range(10, 20) for j in range(10, 20) if j > i and connectivity[i, j] > 0.0]
    damage_idxes = random.sample(idxes, int(len(idxes) * 0.5))
    for (i, j) in damage_idxes:
        connectivity[i, j] *= 0.1
        connectivity[j, i] *= 0.1
    return connectivity

if __name__ == '__main__':
    main()