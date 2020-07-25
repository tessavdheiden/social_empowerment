import argparse
import numpy as np
from numpy import load
from pathlib import Path
import matplotlib.pyplot as plt
from estimate_empowerment import estimate_empowerment_from_positions
from multiagent.scenarios.simple_spread import MDP as spreadMDP
from algorithms.maddpg import MADDPG
import os
from PIL import Image, ImageSequence

def plot_snapshots(config):
    model_path = (Path('./models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental) if config.incremental is not None else model_path / 'model.pt'
    stats_path = model_path.parent / 'stats'
    gif_path = model_path.parent / 'gifs'

    # Load data to be plotted
    all_infos = load(f'{stats_path}/all_infos.npy')
    all_images = Image.open(f'{gif_path}/0_{config.ep_num}.gif')
    all_positions = load(f'{stats_path}/all_positions.npy')

    t = np.arange(len(all_infos[config.ep_num]))
    fig = plt.figure(figsize=(35, 20))
    grid = plt.GridSpec(nrows=3, ncols=25, wspace=0.1, hspace=0.3, figure=fig)

    ax = plt.subplot(grid[0, :])
    for i, frame in enumerate(ImageSequence.Iterator(all_images)):
        if i >= len(t): break
        ax = plt.subplot(grid[0, i])
        ax.imshow(frame), ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel(f"{i}")

    ax = plt.subplot(grid[1, :])
    rewards = all_infos[config.ep_num, :, 0, 0]
    ax.plot(t, rewards, 'b')
    ax.set_xlabel('time (s)'), ax.set_ylabel('Reward', color='b')

    if config.with_empowerment:
        n_agents = 3
        mdp = spreadMDP(n_agents=n_agents, dims=(3, 3), n_step=1)
        E = [estimate_empowerment_from_positions(p, Tn=mdp.Tn, locations=mdp.configurations) for p in all_positions[config.ep_num]]
        ax2 = ax.twinx()
        ax2.plot(t, E, 'r')
        ax2.set_ylabel('Empowerment', color='r')

    if config.with_speech:
        ax2 = plt.subplot(grid[2, :])
        ax2.plot(t, all_infos[config.ep_num, :, 0, 1], c='b')
        ax2.scatter(t, all_infos[config.ep_num, :, 0, 1], c='b')
        ax2.set_ylim(0, 5)
        ax2.set_ylabel('Message')
        ax2.set_xlabel('time (s)')
        ax2.scatter(t, all_infos[config.ep_num, :, 1, 1], c='r')
        ax2.grid(b=True)


    plt.savefig(f'{stats_path}/rewards_episode_{config.ep_num}.png')

    fig, ax = plt.subplots(1, 1)
    alphas = np.linspace(0, 1, all_images.n_frames)

    for i, frame in enumerate(ImageSequence.Iterator(all_images)):
        if i % 3 != 0: continue
        ax.imshow(frame, alpha=alphas[-i]), ax.set_xticks([]), ax.set_yticks([])
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
    parser.add_argument("--with_speech",
                        action='store_true')
    config = parser.parse_args()
    plot_snapshots(config)
