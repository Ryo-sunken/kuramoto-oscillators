import random
import numpy as np
import numpy.linalg as la
import networkx as nx
from collections import deque

def eig_max(A):
    return np.max(la.eig(A)[0])

def holme_kim_graph(n, n0, k, p):
    G = nx.complete_graph(n0)
    for i in range(n-n0):
        target = random.choices(list(G.nodes), weights=dict(nx.degree(G)).values())[0]
        G.add_edge(n0+i, target)
        for _ in range(k-1):
            if np.random.rand() < p:
                nodes = list(nx.all_neighbors(G, target))
                nodes.remove(n0+i)
                cluster_target = random.choice(nodes)
                G.add_edge(n0+i, cluster_target)
            else:
                nodes = list(G.nodes)
                nodes.remove(n0+i)
                nodes.remove(target)
                weights = list(dict(nx.degree(G)).values())
                weights.pop(n0+i)
                weights.pop(target)
                another_target = random.choices(nodes, weights=weights)[0]
                G.add_edge(n0+i, another_target)
    return G

def uniform_weighted_adjacency(G, lower, upper):
    A = nx.to_numpy_array(G)
    for i in range(A.shape[0]):
        for j in range(A.shape[1] - i):
            if A[i, i+j] > 0:
                A[i, i+j] = np.random.rand() * (upper - lower) + lower
                A[i+j, i] = A[i, i+j]
    return A

def weighted_complete_adjacency(n, l, u):
    return uniform_weighted_adjacency(nx.complete_graph(n), l, u)

def weighted_watts_strogatz_adjacency(n, k, p, l, u):
    return uniform_weighted_adjacency(nx.watts_strogatz_graph(n, k, p), l, u)

def weighted_holme_kim_adjacency(n, n0, k, p, l, u):
    return uniform_weighted_adjacency(holme_kim_graph(n, n0, k, p), l, u)

def weighted_laplacian_matrix(A):
    return np.diag(np.sum(A, axis=0)) - A

def adjecency_matrix_from_laplacian(L):
    return -L + np.diag(np.diag(L))

def laplacian_matrix_of_complete_graph(wk):
    return -np.outer(np.ones(wk.shape[0]), wk) + np.eye(wk.shape[0])

def spanning_tree_incidence(A):
    num = A.shape[0]
    B = np.zeros((num, num-1))
    edges = []
    seen_flg = [0] * num
    stack = deque([])
    seen_flg[0] = 1
    stack.append(0)
    while stack:
        pop_flag = True
        for i in range(num):
            if A[stack[-1], i] > 0 and seen_flg[i] == 0:
                edges.append((stack[-1], i))
                stack.append(i)
                seen_flg[i] = 1
                pop_flag = False
                break
        if pop_flag: stack.pop()
    
    for e, edge in zip(range(len(edges)), edges):
        B[edge[0], e] = -1
        B[edge[1], e] = 1
    return B