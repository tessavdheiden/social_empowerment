import argparse
import numpy as np
from numpy import load
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt


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


def plot_data(y, alg_name, color, ax, subsample=10):
    y = y[:len(y)-(len(y)%subsample)]
    mean = np.mean(y.reshape(-1, subsample), axis=1)
    std = np.std(y.reshape(-1, subsample), axis=1)
    ax.plot(np.arange(mean.shape[0]), mean, color=color, label=alg_name)
    ax.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, color=color, alpha=0.2)
    ax.grid('on')
    ax.set_xlabel('TrainSteps')


def plot_training_curve(config):
    plt.rc('font', family='serif')
    model_path = Path('./models') / config.env_id
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    color_list = {'maddpg': [0.65, 0.15, 0.15], 'ddpg': [0.15, 0.65, 0.15], 'maddpg+ve3': [0.15, 0.15, 0.65],
                  'maddpg+si': [0.65, 0.15, 0.65]}

    curve_name = 'rew_loss' # 'pol_loss'  'vf_loss'
    agent_num = 0
    axis = 1
    curve = {}
    for r, d, f in os.walk(model_path):
        for file in f:
            if file.endswith(".json"):
                algorithm_name = r.split('/')[2]
                run = r.split('/')[3][-1]

                if algorithm_name not in color_list: continue
                file_path = os.path.join(r, file)
                y = load_data(file_path, name=curve_name, agent_num=agent_num)
                if algorithm_name in curve:
                    curve[algorithm_name] += y
                    curve[algorithm_name] /= 2
                else:
                    curve[algorithm_name] = y

    for (algorithm_name, y) in curve.items():
        plot_data(y, alg_name=algorithm_name, color=color_list[algorithm_name], ax=ax[axis], subsample=500)
    ax[axis].set_ylabel('AvarageReturn', fontsize=11)
    #ax[axis].legend()
    #ax[1].set_ylim([-8, -5])


    ax[0].set_ylabel('PolicyLoss', fontsize=11)
    ax[2].set_ylabel('CriticLoss', fontsize=11)
    ax[0].legend()


    plt.tight_layout()
    plt.savefig(model_path / f'learning_curve_agent{agent_num}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    config = parser.parse_args()
    plot_training_curve(config)