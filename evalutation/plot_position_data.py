import argparse
import numpy as np
from numpy import load
from pathlib import Path
import matplotlib.pyplot as plt

def plot(config):
    model_path = (Path('models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_plots:
        plot_path = model_path.parent / 'plots'
        plot_path.mkdir(exist_ok=True)

    stats_path = model_path.parent / 'stats'
    all_infos = load(f'{stats_path}/all_infos.npy')
    all_actions = load(f'{stats_path}/all_actions.npy')
    all_positions = load(f'{stats_path}/all_positions.npy')
    all_obs = load(f'{stats_path}/all_observations.npy')

    plt.rc('font', family='serif')
    n_episodes, episode_length, n_agents, dim_a = all_actions.shape
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    if config.save_plots:
        for ep_i in range(n_episodes):
            obs_speaker = all_obs[ep_i, 0, 4:16]
            obstacles = obs_speaker.reshape(-1, 2)
            goal = all_obs[ep_i, 0, 2:4]

            a = 0
            m = np.zeros(5)
            m[ep_i]=1
            ax.plot(all_positions[ep_i, :, a, 0], all_positions[ep_i, :, a, 1],label=f'$m_{ep_i}^0=${1}')
            # ax.scatter(goal[0],goal[1], marker='*', label='goal', c='r')
            # ax.scatter(obstacles[:,0], obstacles[:, 1], marker='X', c='k', label='obstacles')
            ax.axis([-1,1,-1,1])
            ax.set_xlabel('x-coordinate', fontsize=16)
            ax.set_ylabel('y-coordinate', fontsize=16)
            # ax.set_title(config.model_name)
            # ax.legend(loc='lower right')
            ax.set_xticks([])
            ax.set_yticks([])
            # f.savefig(f'{plot_path}/stem_{config.model_name}_ep_{ep_i}_agent_listener.png')

            # f, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
            a = 0
            data = all_actions[ep_i, :, a]
            y = np.argmax(data,axis=1)
            # ax.stem(np.arange(episode_length), y)
        # ax.set_ylim([0, dim_a])
        # ax.set_xlabel('Time', fontsize=11)
        # ax.set_ylabel('Symbol', fontsize=11)
        # ax.set_title(config.model_name)
        plt.tight_layout()
        f.savefig(f'{plot_path}/stem_{config.model_name}_ep_{ep_i}_agent_speaker.png')
        plt.close(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_plots", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=50, type=int)
    parser.add_argument("--episode_length", default=10, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--mixed_policies", action="store_true")

    config = parser.parse_args()

    plot(config)