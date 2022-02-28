import argparse
import numpy as np
from numpy import load
from pathlib import Path
import matplotlib.pyplot as plt

plt.rc('font', family='serif')


def print_table(config):
    model_path = (Path('../models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_plots:
        plot_path = model_path.parent / 'plots'
        plot_path.mkdir(exist_ok=True)

    stats_path = model_path.parent / 'stats'
    all_infos = load(f'{stats_path}/all_infos.npy')
    all_positions =  load(f'{stats_path}/all_positions.npy')
    all_communications = load(f'{stats_path}/all_communications.npy')
    all_obs = load(f'{stats_path}/all_observations.npy')

    plt.rc('font', family='serif')
    n_episodes,episode_length,n_speaking_agents, dim_c = all_communications.shape
    for a in range(n_speaking_agents):
        if config.save_plots:
            f, ax = plt.subplots(1, 1, figsize=(3, 3))

            bottom_data = all_communications[0, :, a].sum(0) / episode_length
            # ax.bar(np.arange(dim_c), bottom_data, color=all_obs[0,0,0:3])
            ax.bar(np.arange(dim_c), bottom_data)
            for ep_i in range(1, n_episodes):
                goal_color = all_obs[ep_i,0,0:3]
                data = all_communications[ep_i, :, a].sum(0) / episode_length
                # ax.bar(np.arange(dim_c), data, bottom=bottom_data,color=goal_color)
                ax.bar(np.arange(dim_c), data, bottom=bottom_data)
                bottom_data += data
                # ax.set_ylim([0,1])
            ax.set_xlabel('Symbol', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(config.model_name)
            plt.tight_layout()
            f.savefig(f'{plot_path}/communications_{config.model_name}.png')
            plt.close(f)

    # for a in range(n_speaking_agents):
    #     if config.save_plots:
    #         f, ax = plt.subplots(1, 1, figsize=(3, 3))
    #         for ep_i in range(n_episodes):
    #             data = all_communications[ep_i, :, a]
    #             ax.stem(np.arange(episode_length), data)
    #             ax.set_ylim([0,len(dim_c)])
    #         ax.set_xlabel('Time', fontsize=11)
    #         ax.set_ylabel('Symbol', fontsize=11)
    #         plt.tight_layout()
    #         f.savefig(f'{plot_path}/stem_{config.model_name}_ep_{ep_i}.png')
    #         plt.close(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--save_plots", action="store_true",
                        help="Saves plot of all episodes into model directory")
    config = parser.parse_args()
    print_table(config)