import argparse
import numpy as np
from numpy import load
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])


def load_data(file_path, name):
    cast = lambda x: np.array(x)
    with open(file_path) as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key.split('/')[-1] == name:
                d = cast(value)
                return d[:, 2]


def plot_data(y, alg_name, color, ax):
    mean = np.mean(y.reshape(-1, 10), axis=1)
    std = np.std(y.reshape(-1, 10), axis=1)

    ax.plot(np.arange(mean.shape[0]), mean, color=color, label=alg_name)
    ax.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, color=color, alpha=0.1)
    ax.grid('on')
    ax.set_xlabel('TrainSteps')
    ax.legend()


def plot_training_curve(config):
    plt.rc('font', family='serif')
    model_path = Path('./models') / config.env_id
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    n_files = 0
    for r, d, f in os.walk(model_path):
        for file in f:
            if file.endswith(".json"):
                file_path = os.path.join(r, file)
                y1 = load_data(file_path, name='pol_loss')
                plot_data(y1,  alg_name=r.split('/')[2], color=colors[n_files], ax=ax[0])
                n_files += 1

    ax[0].set_title('StationaryAgent AvarageReturn', fontsize=11)
    ax[1].set_title('Non-StationaryAgent AvarageReturn', fontsize=11)

    plt.tight_layout()
    plt.savefig(model_path / 'learning_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    config = parser.parse_args()
    plot_training_curve(config)