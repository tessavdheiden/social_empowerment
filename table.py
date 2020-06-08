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
    if config.env_id == 'simple_spread':
        n_agents = 3
        collisions = np.any(all_infos[:, :, 1] > 1, axis=1).mean()
        avg_dist = all_infos[:, -1, 2].mean() * n_agents

        print(f'collisions = {collisions:.3f} maximal one per episode and averaged over {len(all_infos)} episodes')
        print(f'episodes in collision = {np.argwhere(np.any(all_infos[:, :, 1] > 1, axis=1)).reshape(-1)}')
        print(f'min_dist = {avg_dist:.3f} minimal distance at end of episode and averaged over {len(all_infos)} episodes')


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