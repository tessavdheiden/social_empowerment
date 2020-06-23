import numpy as np
from functools import reduce
import itertools

from multiagent.scenarios.transition_utils import _location_to_index, _cell_to_index
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

def estimate_empowerment_from_landmark_positions(config, Tn, locations):
    return empowerment(Tn, det=1., n_step=1, state=config)



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

