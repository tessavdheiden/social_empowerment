import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run(config):
    model_path = (Path('models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs' if not config.mixed_policies else model_path.parent / 'gifs_mixed'
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

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        # env.agents[1].state.p_pos = np.array([0., 0.])
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False) if not obs[i].ndim == 4 else Variable(torch.Tensor(obs[i]), requires_grad=False)
                         for i in range(maddpg.nagents)]

            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            # actions[0] = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            obs, rewards, dones, infos = env.step(actions)

            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
                # frames.append(env.world.viewers[0].render(return_rgb_array = True)) uncomment if local views visible
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')

        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--mixed_policies", action="store_true")

    config = parser.parse_args()

    run(config)