import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
import argparse

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def parse_arguments():
    parser = argparse.ArgumentParser(description='シミュレーション結果を描画')
    parser.add_argument('folder_name',                                                                                                               help='結果ファイルが入っているフォルダー名')
    parser.add_argument('network_file_name', help='ネットワークパラメータが入っているファイル名')
    parser.add_argument('control_file_name', help='制御パラメータが入っているファイル名')
    parser.add_argument('file_number',       type=int,                                                                                               help='何番目のファイルを読み込むか')
    parser.add_argument('--lim',             required=False, type=float, nargs=2,                default=[0, 5],                                     help='時間軸の範囲')
    parser.add_argument('--style',           required=False, choices=['wave', 'order', 'phase', 'max'], default='max',                                     help='結果の描画形式')

    return parser.parse_args()

def load_network_param():
    path = 'data/' + args.folder_name + '/param/network/' + args.network_file_name + '.json'
    with open(path, mode='r', encoding='utf-8') as file:
        param = json.load(file)
    return param

def load_result(network_param):
    path = 'data/' + args.folder_name + '/result/' + args.network_file_name + '/' + args.control_file_name
    files = os.listdir(path)
    names = ['time'] + ['phase' + str(i) for i in range(1, int(network_param['state_dim']+1))]
    return pd.read_csv(path + '/' + files[args.file_number], delimiter=' ', names=names)

def create_fig():
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fsize = 22
    cmap = plt.get_cmap('tab10')
    ax.set_xticks(np.arange(0.0, 50.5, 1.0))

    if args.style == 'wave':
        k = 0
        idx_sum = 0
        for i in range(param['state_dim']):
            phase = np.array(result['phase'+str(i+1)])
            if i+1 == idx_sum + param['cluster_nodes_num'][k]:
                ax.plot(t, np.sin(phase), linewidth=1.5, color=cmap(k), label='Cluster'+str(k+1))
                idx_sum += param['cluster_nodes_num'][k]
                k += 1
            else:
                ax.plot(t, np.sin(phase), linewidth=1.5, color=cmap(k))
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_ylabel(r'$\sin(\theta)$', fontsize=fsize)
    elif args.style == 'order':
        idx_sum = 0
        for k in range(len(param['cluster_nodes_num'])):
            data = []
            for i in range(idx_sum+1, idx_sum+param['cluster_nodes_num'][k]+1):
                phase = np.array(result['phase'+str(i)])
                data.append(list(phase))
            data = np.array(data)
            order_data = np.sqrt(np.power(np.sum(np.cos(data), 0), 2) + np.power(np.sum(np.sin(data), 0), 2)) / param['cluster_nodes_num'][k]

            ax.plot(t, order_data, linewidth=1.2, color=cmap(k), label='Cluster' + str(k+1))

            idx_sum += param['cluster_nodes_num'][k]

        ax.set_ylim(0.9, 1.01)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_ylabel(r'$r$', fontsize=fsize)
    elif args.style == 'phase':
        k = 0
        idx_sum = 0
        for i in range(param['state_dim']):
            if i+1 == idx_sum + param['cluster_nodes_num'][k]:
                ax.plot(t, result['phase'+str(i+1)], linewidth=1, color=cmap(k), label='Cluster'+str(k+1))
                idx_sum += param['cluster_nodes_num'][k]
                k += 1
            else:
                ax.plot(t, result['phase'+str(i+1)], linewidth=1, color=cmap(k))
        ax.set_ylim(0, 2*np.pi)
        ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ax.set_ylabel(r'$\theta$', fontsize=fsize)
    elif args.style == 'max':
        idx_sum = 0
        for k in range(len(param['cluster_nodes_num'])):
            data = []
            for i in range(idx_sum+1, idx_sum+param['cluster_nodes_num'][k]+1):
                phase = np.array(result['phase'+str(i)])
                data.append(list(phase))
            data = np.array(data)
            B_comp = np.zeros((param['cluster_nodes_num'][k], int(param['cluster_nodes_num'][k] * (param['cluster_nodes_num'][k] - 1) / 2)))
            edge = 0
            for i in range(param['cluster_nodes_num'][k]):
                for j in range(i + 1, param['cluster_nodes_num'][k]):
                    B_comp[i, edge] = 1
                    B_comp[j, edge] = -1
                    edge += 1
            phase_diff = np.abs(B_comp.T @ data)
            for i in range(phase_diff.shape[0]):
                for j in range(phase_diff.shape[1]):
                    if phase_diff[i, j] > np.pi:
                        phase_diff[i, j] = 2 * np.pi - phase_diff[i, j]
            max_data = np.max(phase_diff, axis=0)
            ax.plot(t, max_data, linewidth=1.2, color=cmap(k), label='Cluster' + str(k+1))
            idx_sum += param['cluster_nodes_num'][k]

        ax.set_ylim(0, np.pi)
        ax.set_yticks([0, 1.0, 2.0, 3.0])
        ax.set_ylabel(r'Max phase difference', fontsize=fsize)


    ax.set_xlim(args.lim[0], args.lim[1])
    ax.set_xlabel('Time $t$', fontsize=fsize)
    ax.tick_params(axis='both', labelsize=fsize)
    ax.legend(fontsize=fsize, loc='lower right', framealpha=1)

    plt.tight_layout()

if __name__ == '__main__':
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True

    args   = parse_arguments()
    param  = load_network_param()
    result = load_result(param)
    t      = result['time']

    create_fig()

    plt.show()