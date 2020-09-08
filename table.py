import argparse
import numpy as np
from numpy import load
from pathlib import Path


def print_table(config):
    model_path = (Path('./models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    stats_path = model_path.parent / 'stats'
    all_infos = load(f'{stats_path}/all_infos.npy')
    if config.env_id == 'simple_order':
        collisions = np.any(all_infos[:, :, :, 1] > 1, axis=1).mean()
        avg_dist = all_infos[:, :, :, 2].mean()
        reward = all_infos[:, :, :, 0].mean()
        fin_dist = all_infos[:, -5:, 1, 1].mean()
        target_reach = all_infos[:, :, :, 3].mean()

        print(f'reward: \t obs_hit: \t avg_dist:  \t fin_dis:  \t target_reach: ')
        print(f'\t {reward:.3f}\t {collisions:.3f} \t {avg_dist:.3f} \t {fin_dist:.3f} \t {target_reach:.3f}')

    elif config.env_id == 'simple_speaker_listener3':
        ep, t, n_agents, metrics = all_infos.shape
        avg_dist = all_infos[:, :, 1, 1].mean()
        fin_dist = all_infos[:, -5:, 1, 1].mean()
        target_reach = np.all(all_infos[:, -5:, 1, 1] < .1, axis=1).mean()
        reward = all_infos[:, :, 1, 0].mean()

        collisions = np.any(all_infos[:, :, 1, 3], axis=1).mean()
        alternating_frequency = np.array(np.clip(np.abs(np.diff(all_infos[:, :, 0, 2], axis=1)), 0, 1)).mean()
        distinct_token = np.array(list(map(len, np.unique(all_infos[:, -5:, 0, 2], axis=1)))).mean()

        print(f'reward: \t obs_hit: \t avg_dist:  \t fin_dis:  \t target_reach: \t distinct_token: \t alternating_frequency:')
        print(f'\t {reward:.3f}\t {collisions:.3f} \t {avg_dist:.3f} \t {fin_dist:.3f} \t {target_reach:.3f} \t {distinct_token:.3f} \t {alternating_frequency:.3f}')

    elif config.env_id == 'simple_car_pixels':
        reward = all_infos[:, :, 1, 0].mean()
        collisions = np.any(all_infos[:, :, 0, 1], axis=1).mean()
        off_road = np.any(all_infos[:, :, 0, 2], axis=1).mean()

        print(f'reward: \t collisions: \t off_road: ')
        print(f'\t {reward:.3f}\t {collisions:.3f} \t {off_road:.3f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    config = parser.parse_args()
    print_table(config)