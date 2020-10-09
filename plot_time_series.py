import argparse
import numpy as np
from numpy import load
from pathlib import Path
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
from PIL import Image, ImageSequence
from scipy.ndimage.filters import gaussian_filter1d

colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15],
                   [0.15, 0.15, 0.15], [0.65, 0.65, 0.65]])


def load_data(config):
    model_path = (Path('./models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental) if config.incremental is not None else model_path / 'model.pt'
    stats_path = model_path.parent / 'stats'
    gif_path = model_path.parent / 'gifs'

    # Load data to be plotted
    all_infos = load(f'{stats_path}/all_infos.npy')
    all_images = Image.open(f'{gif_path}/0_{config.ep_num}.gif')
    all_positions = load(f'{stats_path}/all_positions.npy')
    return all_infos, all_images, all_positions, stats_path


def plot_snapshots_grid(config, all_infos, all_images, all_positions, stats_path):
    t = np.arange(len(all_infos[config.ep_num]))
    r = all_infos[config.ep_num, :, 0, 0]
    r_smooth = gaussian_filter1d(r, sigma=2)

    fig = plt.figure(figsize=(65, 20))
    n_cols = all_images.n_frames
    grid = plt.GridSpec(nrows=3, ncols=n_cols, wspace=0.1, hspace=0.3, figure=fig)

    # Snapshots
    for i, frame in enumerate(ImageSequence.Iterator(all_images)):
        grid_ax = plt.subplot(grid[0, i])
        grid_ax.imshow(frame), grid_ax.set_xticks([]), grid_ax.set_yticks([])
        grid_ax.set_xlabel(f"t={i}")

    # Rewards
    grid_ax = plt.subplot(grid[1, :])
    grid_ax.plot(t, r_smooth, linestyle='solid', linewidth=5, color=colors[0])
    grid_ax.set_xlabel('time (s)'), grid_ax.set_ylabel('Reward', color=colors[0])
    grid_ax.set_xticks(np.arange(n_cols))
    grid_ax.xaxis.grid()

    plt.savefig(f'{stats_path}/rewards_snapshots_episode_{config.ep_num}.png')


def plot_all_in_one(config, all_infos, all_images, all_positions, stats_path):
    fig, ax = plt.subplots(1, 1)

    for i, frame in enumerate(ImageSequence.Iterator(all_images)):
        if i % 20 != 0: continue
        ax.imshow(frame, alpha=.1), ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel(f"{i}")

    plt.savefig(f'{stats_path}/all_in_one_{config.ep_num}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("ep_num", default=2, type=int)
    parser.add_argument("--with_empowerment",
                        action='store_true')
    parser.add_argument("--grid", default=0, type=int)
    parser.add_argument("--with_speech",
                        action='store_true')
    config = parser.parse_args()

    plot_fn = plot_all_in_one

    for i in range(10):
        config.ep_num = i
        all_infos, all_images, all_positions, stats_path = load_data(config)

        plot_fn(config, all_infos, all_images, all_positions, stats_path)

