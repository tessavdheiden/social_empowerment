import argparse
import numpy as np
from pathlib import Path
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15],
                   [0.15, 0.15, 0.15], [0.65, 0.65, 0.65], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]])


def load_data(file_path, name, agent_num=0):
    cast = lambda x: np.array(x)
    with open(file_path) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key.split('/')[-1] == name and int(key.split('/')[-3][-1]) == agent_num:
                d = cast(value)
                return d[:, 2]


def plot_training_curve(config):
    model_path = Path('./models') / config.env_id

    names = ['baseline', 'empowerment', 'random', 'maddpg-rnn']
    data = defaultdict(np.array)

    curve_name = 'rew_loss' # 'pol_loss'  'vf_loss'
    agent_num = 0
    axis = 1
    n_points = 100

    # find files
    for r, _, files in os.walk(model_path):
        for f in files:
            if f.endswith(".json"):
                name = r.split('/')[2]
                if name not in names: continue
                y = load_data(os.path.join(r, f), name=curve_name, agent_num=agent_num)
                data[name] = y

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    data = dict(sorted(data.items()))

    colors = ['r', 'g', 'b', 'c']
    for i, (name, data) in enumerate(data.items()):
        y_smoothed = gaussian_filter1d(data, sigma=40)
        ax[axis].plot(y_smoothed, colors[i], label=name)
        # ax[axis].plot(y, color, alpha=0.1)
        ax[axis].set_xlabel('Training steps')
        ax[axis].legend()

    ax[axis].set_ylabel('Avarage return', fontsize=11)
    ax[axis].set_ylim([-4, 1])
    # ax[0].set_ylabel('PolicyLoss', fontsize=11)
    # ax[2].set_ylabel('CriticLoss', fontsize=11)
    plt.tight_layout()
    plt.savefig(model_path / f'learning_curve_agent{agent_num}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    config = parser.parse_args()
    plot_training_curve(config)