import argparse
import torch
import time
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from numpy import save
import matplotlib.pyplot as plt

def run(config):
    model_path = (Path('../models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    gif_path = model_path.parent / 'stats' if not config.mixed_policies else model_path.parent / 'stats_mixed'
    gif_path.mkdir(exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.mixed_policies:
        maddpg = MADDPG.init_from_directory(Path('../models') / config.env_id / config.model_name)
    else:
        maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, benchmark=True, discrete_action=maddpg.discrete_action)
    env.seed(config.seed)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    all_infos = np.empty((config.n_episodes, config.episode_length, maddpg.nagents, 10))
    n_movable_agents = sum([1 if a.movable else 0 for a in env.agents])
    n_speaking_agents = sum([0 if a.silent else 1 for a in env.agents])
    all_positions = np.zeros((config.n_episodes, config.episode_length, n_movable_agents, env.world.dim_p))
    all_communications = np.zeros((config.n_episodes, config.episode_length, n_speaking_agents, env.world.dim_c))
    all_actions = np.zeros((config.n_episodes, config.episode_length, len(env.agents), env.world.dim_c))
    obs_space = sum([obsp.shape[0] for obsp in env.observation_space])
    all_obs = np.zeros((config.n_episodes, config.episode_length, obs_space))

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        # env.agents[1].state.p_pos = np.array([0., 0.])
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False) if not obs[i].ndim == 4 else Variable(torch.Tensor(obs[i]), requires_grad=False)
                         for i in range(maddpg.nagents)]

            all_positions[ep_i, t_i] = env.get_positions()
            all_communications[ep_i, t_i] = env.get_communications()
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            # actions[0] = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            # actions[0][ep_i] = 1.
            obs, rewards, dones, infos = env.step(actions)

            all_actions[ep_i, t_i, :, :] = actions
            all_obs[ep_i, t_i, :] = np.concatenate(np.asarray(obs))

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            if len(np.array(infos['n']).shape) < 4:
                all_infos[ep_i, t_i, :, :len(infos['n'][-1])] = np.array(infos['n'])

    env.close()

    if config.save_stats:
        stats_path = model_path.parent / 'stats' if not config.mixed_policies else model_path.parent / 'stats_mixed'
        stats_path.mkdir(exist_ok=True)
        save(f'{stats_path}/all_infos.npy', all_infos)
        save(f'{stats_path}/all_positions.npy', all_positions)
        save(f'{stats_path}/all_communications.npy', all_communications)
        save(f'{stats_path}/all_actions.npy', all_actions)
        save(f'{stats_path}/all_observations.npy', all_obs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_stats", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=50, type=int)
    parser.add_argument("--episode_length", default=10, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--mixed_policies", action="store_true")

    config = parser.parse_args()

    run(config)