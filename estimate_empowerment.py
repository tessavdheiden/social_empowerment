import numpy as np
from numpy import load
from pathlib import Path
from functools import reduce
import itertools

from algorithms.compute_transition import compute_transition, compute_transition_nstep, _location_to_index, _index_to_cell, _cell_to_index
from algorithms.info_theory import blahut_arimoto


def estimate_empowerment_from_positions(ps, Tn, locations, dims=(3, 3)):
    cells = np.array([_positions_to_cell(pose, x_lim_out=(0, dims[0]), y_lim_out=(0, dims[1])) for pose in
                      ps]).reshape(-1, 2)

    if (not _cells_in_collision(cells)) and (not _cells_outside_bounds(cells, dims)):
        ls = np.array([_cell_to_index(c, dims=dims) for c in cells])
        if not _locations_in_collision(ls):
            ss = _location_to_index(locs=ls, locations=locations)
            return empowerment(Tn, det=1., n_step=1, state=ss)
    return 0.


def _positions_to_cell(ps, x_lim_in=(-1, 1), y_lim_in=(-1, 1), x_lim_out=(0, 3), y_lim_out=(0, 3)):
    def transform(p, lim_in, lim_out):
        p_out = (p - lim_in[0]) / (lim_in[1] - lim_in[0])
        return int(np.clip(p_out * (lim_out[1] - lim_out[0]) + lim_out[0], a_min=lim_out[0], a_max=lim_out[1]))

    return np.array([transform(ps[1], y_lim_in, y_lim_out), transform(ps[0], x_lim_in, x_lim_out)])

def _cells_in_collision(cells):
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i == j: continue
            if np.array_equal(cells[i], cells[j]):
                return True
    return False

def _locations_in_collision(locations):
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j: continue
            if np.array_equal(locations[i], locations[j]):
                return True
    return False

def _cells_outside_bounds(cells, dims):
    for cell in cells:
        if np.any(cell < np.zeros(2)) or np.any(cell >= dims):
            return True
    return False



def empowerment(T, det, n_step, state, n_samples=1000, epsilon=1e-6):
    """
    Compute the empowerment of a state in a grid world
    T : numpy array, shape (n_states, n_actions, n_states)
        Transition matrix describing the probabilistic dynamics of a markov decision process
        (without rewards). Taking action a in state s, T describes a probability distribution
        over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of
        landing in state s' after taking action a in state s. The indices may seem "backwards"
        because this allows for convenient matrix multiplication.
    det : bool
        True if the dynamics are deterministic.
    n_step : int
        Determines the "time horizon" of the empowerment computation. The computed empowerment is
        the influence the agent has on the future over an n_step time horizon.
    n_samples : int
        Number of samples for approximating the empowerment in the deterministic case.
    state : int
        State for which to compute the empowerment.
    """
    n_states, n_actions, _ = T.shape
    if det==1:
        # only sample if too many actions sequences to iterate through
        if n_actions ** n_step < 5000:
            nstep_samples = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
        else:
            nstep_samples = np.random.randint(0, n_actions, [n_samples, n_step])
        # fold over each nstep actions, get unique end states
        tmap = lambda s, a: np.argmax(T[:, a, s])
        seen = set()
        for i in range(len(nstep_samples)):
            aseq = nstep_samples[i, :]
            seen.add(reduce(tmap, [state, *aseq]))
        # empowerment = log # of reachable states
        return np.log2(len(seen))
    else:
        nstep_actions = list(itertools.product(range(n_actions), repeat=n_step))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))
        q_x = _rand_dist((Bn.shape[1],))
        return blahut_arimoto(Bn[:, :, state], q_x, epsilon=epsilon)


def _rand_dist(shape):
    """ define a random probability distribution """
    P = np.random.rand(*shape)
    return _normalize(P)


def _normalize(P):
    """ normalize probability distribution """
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt


    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    config = parser.parse_args()
    model_path = (Path('./models') / config.env_id / config.model_name / ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    stats_path = model_path.parent / 'stats'
    all_positions = load(f'{stats_path}/all_positions.npy')
    n_episodes = len(all_positions)
    episode_length = len(all_positions[0])
    fig, ax = plt.subplots(nrows=2, ncols=episode_length, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    dims = (3, 3)
    Tn = compute_transition_nstep(T=compute_transition(n_agents=3, dims=dims), n_step=1)
    locations = np.array(list(itertools.permutations(np.arange(dims[0]*dims[1]), 3)), dtype='uint8')
    for ep_i in range(n_episodes):
        if ep_i != 1: continue
        for t_i in range(0, episode_length):
            positions = all_positions[ep_i][t_i]
            [ax[0, t_i].add_artist(plt.Circle((p[0], p[1]), 0.15)) for p in positions]
            ax[0, t_i].axis('square')
            ax[0, t_i].set(xlim=(-1, 1), ylim=(-1, 1))

            cells = np.array([_positions_to_cell(position, x_lim_out=(0, dims[0]), y_lim_out=(0, dims[1])) for position in positions]).reshape(-1, 2)

            [ax[1, t_i].add_artist(plt.Circle((p[1]+.5, p[0]+.5), 0.5)) for p in cells]
            ax[1, t_i].axis('square')
            ax[1, t_i].set(xlim=(0, dims[1]), ylim=(0, dims[0]))

            E = estimate_empowerment_from_positions(positions, Tn=Tn, locations=locations, dims=dims)
            ax[1, t_i].set_xlabel(f'E={E:.2f}')

    plt.savefig('tmp.png')
