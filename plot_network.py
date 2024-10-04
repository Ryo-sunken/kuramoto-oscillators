import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='ネットワークとパラメータの図示')
    parser.add_argument('folder_name',                                                                    help='パラメータファイルが入っているフォルダー名')
    parser.add_argument('file_name',                                                                      help='開くパラメータファイル名（.json除く）')
    parser.add_argument('-f', '--freq', required=False, action='store_true',                              help='周波数のグラフを表示するフラグ')
    parser.add_argument('--coords',     required=False, choices=['line', 'cylinder'], default='cylinder', help='ノードの配置方法')

    return parser.parse_args()

def load_param():
    path = 'data/' + args.folder_name + '/param/network/' + args.file_name + '.json'
    with open(path, mode='r', encoding='utf-8') as file:
        param = json.load(file)
    return param

def create_coords():
    cluster = np.array(param['cluster_nodes_num'])
    xs = []
    ys = []
    if args.coords == 'line':
        center_x = [0] * cluster.size
        center_y = [1.0 - k / cluster.size * 3.0 for k in range(cluster.size)]
        for k in range(cluster.size):
            xs.append([-0.8 + i / cluster[k] * 2.0 + center_x[k] for i in range(cluster[k])])
            ys.append([0 + center_y[k]] * cluster[k])
    elif args.coords == 'cylinder':
        center_x = [0] * cluster.size
        center_y = [1.0 - k / cluster.size * 3.0 for k in range(cluster.size)]
        for k in range(cluster.size):
            xs.append([0.8 * np.cos(i / cluster[k] * 2 * np.pi + 0.2) + center_x[k] for i in range(cluster[k])])
            ys.append([0.4 * np.sin(i / cluster[k] * 2 * np.pi + 0.2) + center_y[k] for i in range(cluster[k])])
    x = [item for row in xs for item in row]
    y = [item for row in ys for item in row]
    return (xs, ys, x, y)

def create_edges():
    edges = []
    max_weight = np.max(connectivity)
    for i in range(connectivity.shape[0]):
        for j in range(i,connectivity.shape[1]):
            if connectivity[i, j] > 0:
                edges.append((i, j, connectivity[i, j] / max_weight))
    return edges

def create_fig():
    grid = plt.GridSpec(3, 2)
    ax = []
    fig = plt.figure()
    ax.append(fig.add_subplot(grid[:, 0]))
    if args.freq:
        ax.append(fig.add_subplot(grid[1:, 1]))
        ax.append(fig.add_subplot(grid[0, 1]))
    else:
        ax.append(fig.add_subplot(grid[:, 1]))

    fig.set_figwidth(14)
    fig.set_figheight(6)
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[1].invert_yaxis()
    ax[1].tick_params(labelsize=14)
    if args.freq:
        ax[2].set_ylabel('frequency')
    return fig, ax

def plot_network():
    (xs, ys, x, y) = create_coords()
    edges          = create_edges()
    cmap           = plt.get_cmap('tab10')

    cluster = np.array(param['cluster_nodes_num'])
    max_weight = np.max(connectivity)
    for e in edges:
        if (10 <= e[0] and e[0] < 20) and (10 <= e[1] and e[1] < 20) and e[2] < 1.0 / max_weight:
            ax[0].plot([x[e[0]], x[e[1]]], [y[e[0]], y[e[1]]], color='k', linewidth=1, linestyle='dashed', zorder=1)
        else:
            ax[0].plot([x[e[0]], x[e[1]]], [y[e[0]], y[e[1]]], color='k', linewidth=2.5 * e[2], zorder=1)
    for k in range(cluster.size):
        ax[0].scatter(xs[k], ys[k], s=280, color=cmap(k), zorder=2)
    for i in range(len(x)):
        ax[0].text(x[i], y[i], i + 1, color='white', size=12, weight='bold', horizontalalignment='center', verticalalignment='center')

def plot_heatmap():
    im = ax[1].pcolor(connectivity, cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax[1])
    cbar.ax.tick_params(labelsize=12)

def plot_freq():
    x = np.arange(1, len(param['frequency']) + 1, 1)
    ax[2].bar(x, param['frequency'], tick_label=x)
    ax[2].tick_params(labelsize=10)

if __name__ == '__main__':
    #plt.rcParams['ps.useafm'] = True
    #plt.rcParams['pdf.use14corefonts'] = True
    #plt.rcParams['text.usetex'] = True

    args         = parse_arguments()
    param        = load_param()
    fig, ax      = create_fig()
    connectivity = np.array(param['connectivity'])

    plot_network()
    plot_heatmap()
    if args.freq: 
        plot_freq()
        
    #plt.savefig('network_adjcency.pdf', backend='pgf')
    plt.show()