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
    all_actions =  load(f'{stats_path}/all_actions.npy')

    plt.rc('font', family='serif')
    n_episodes,episode_length,n_agents, dim_a = all_actions.shape
    for a in range(n_agents):
        if config.save_plots:
            width = 0.1
            f, ax = plt.subplots(1, 1, figsize=(3, 3))
            bottom_data = all_actions[0, :, a].sum(0) / episode_length
            ax.bar(np.arange(dim_a), bottom_data)
            for ep_i in range(1, n_episodes):
                data = all_actions[ep_i, :, a].sum(0) / episode_length
                ax.bar(np.arange(dim_a), data, bottom=bottom_data)
                bottom_data += data
                # ax.set_ylim([0,1])
            if a == 0:
                ax.set_xlabel('Symbol', fontsize=11)
            else:
                ax.set_xlabel('Action', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            # ax.set_title(config.model_name)
            plt.tight_layout()
            f.savefig(f'{plot_path}/{config.model_name}_agent_{a}_actions.png')
            plt.close(f)

    # for a in range(n_agents):
    #     if config.save_plots:
    #
    #         for ep_i in range(n_episodes):
    #             f, ax = plt.subplots(1, 1, figsize=(3, 3))
    #             data = all_actions[ep_i, :, a]
    #             y = np.argwhere(data.T > 0)[:,0]
    #             ax.stem(np.arange(episode_length), y)
    #             ax.set_ylim([0,dim_a])
    #             ax.set_xlabel('Time', fontsize=11)
    #             ax.set_ylabel('Symbol', fontsize=11)
    #             plt.tight_layout()
    #             f.savefig(f'{plot_path}/stem_{config.model_name}_ep_{ep_i}_agent_{a}.png')
    #             plt.close(f)



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