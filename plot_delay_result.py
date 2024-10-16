import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name')
    parser.add_argument('network_file_name')
    parser.add_argument('control_file_name')
    parser.add_argument('seed_number', type=int)

    return parser.parse_args()

def load_network_param():
    path = 'data/' + args.folder_name + '/param/network/' + args.network_file_name + '.json'
    with open(path, mode='r', encoding='utf-8') as file:
        param = json.load(file)
    return param

def load_result_data(path):
    names = ['time'] + ['phase' + str(i) for i in range(1, int(param['state_dim']+1))]
    with open(path, mode='r', encoding='utf-8') as file:
        result = pd.read_csv(file, delimiter=' ', names=names)
    return result

def create_fig():
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fsize = 22
    cmap = plt.get_cmap('tab10')
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))

    plot_data = []

    for i in range(0, 101):
        # ファイル読み込み
        path = ''
        if i == 0:
            path = 'data/' + args.folder_name + '/result/' + args.network_file_name + '/' + args.control_file_name
            files = os.listdir(path)
            path = path + '/' + files[args.seed_number]
        else:
            path = 'data/' + args.folder_name + '/result/' + args.network_file_name + '/delayed_' + args.control_file_name
            folders = os.listdir(path)
            path = path + '/' + folders[args.seed_number]
            path = path + '/' + str(i) + '.csv'
        result = load_result_data(path)

        # 位相データをオーダーパラメータに変換して最小値を記録
        idx_sum = 0
        min_order = 1.0
        for k in range(len(param['cluster_nodes_num'])):
            data = []
            for i in range(idx_sum + 1, idx_sum + param['cluster_nodes_num'][k] + 1):
                phase = np.array(result[result['time'] >= 80.0]['phase' + str(i)])
                data.append(list(phase))
            data = np.array(data)
            order_data = np.sqrt(np.power(np.sum(np.cos(data), 0), 2) + np.power(np.sum(np.sin(data), 0), 2)) / param['cluster_nodes_num'][k]
            if np.min(order_data) < min_order:
                min_order = np.min(order_data)          
            idx_sum += param['cluster_nodes_num'][k]
        plot_data.append(min_order)

    ax.plot(np.arange(0.0, 1.01, 0.01), plot_data, marker='o')
    ax.set_ylim(0.0, 1.0)
        




if __name__ == '__main__':
    args = parse_arguments()
    param = load_network_param()

    create_fig()

    plt.show()