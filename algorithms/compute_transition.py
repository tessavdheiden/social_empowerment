import numpy as np
from functools import reduce
import itertools

actions = {
    "N": np.array([1, 0]),  # UP
    "S": np.array([-1, 0]),  # DOWN
    "E": np.array([0, 1]),  # RIGHT
    "W": np.array([0, -1]),  # LEFT
    "_": np.array([0, 0])  # STAY
}


def _index_to_cell(s, dims):
    cell = [int(s / dims[1]), s % dims[1]]
    return np.array(cell)


def _cell_to_index(cell, dims):
    return cell[1] + dims[1]*cell[0]


def _location_to_index(locations, locs):
    return np.where(np.all(locations == locs, axis=1))[0][0]

vecmod = np.vectorize(lambda x, y : x % y)

def act(s, a, dims, prob = 1., toroidal=False):
    """ get updated state after action
    s  : state, index of grid position
    a : action
    prob : probability of performing action
    """
    rnd = np.random.rand()
    if rnd > prob:
        a = np.random.choice(list(filter(lambda x : x !=a, actions.keys())))
    state = _index_to_cell(s, dims)

    new_state = state + actions[a]
    # can't move off grid
    if toroidal:
        new_state = vecmod(new_state, dims)
    elif np.any(new_state < np.zeros(2)) or np.any(new_state >= dims):
        return _cell_to_index(state, dims)
    return _cell_to_index(new_state, dims)


def switch_places(locs, locs_new):
    """ return true if locations between agents are switched between successive time steps
    locs: locations at t1 of all agents
    locs_t2: locations at t2 of all agents
    """
    if len(locs) == 2:
        return (locs[0] == locs_new[1]) and (locs[1] == locs_new[0])
    elif len(locs) == 3:
        return ((locs[0] == locs_new[1]) and (locs[1] == locs_new[0])) or \
               ((locs[0] == locs_new[2]) and (locs[2] == locs_new[0])) or \
               ((locs[1] == locs_new[2]) and (locs[2] == locs_new[1]))
    else:
        raise NotImplementedError


def compute_transition(n_agents, dims, det=1.):
    n_locations = dims[0]*dims[1]
    locations = np.array(list(itertools.permutations(np.arange(n_locations), n_agents)))
    a_list = list(itertools.product(actions.keys(), repeat=n_agents))
    n_actions = len(a_list)

    n_configs = len(locations)
    # compute environment dynamics as a matrix T
    T = np.zeros([n_configs, n_actions, n_configs], dtype='uint8')
    # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
    for c, locs in enumerate(locations):
        for i, alist in enumerate(a_list):
            locs_new = [act(locs[j], alist[j], dims) for j in range(n_agents)]

            # any of the agents on same location, do not move
            if len(set(locs_new)) < n_agents or switch_places(locs, locs_new):
                locs_new = locs

            c_new = _location_to_index(locations, locs_new)
            T[c_new, i, c] += det

            if det == 1: continue
            locs_unc = np.array(
                [list(map(lambda x: act(locs[j], x, dims, det), filter(lambda x: x != alist[j], actions.keys())))
                 for j in range(n_agents)]).T
            assert locs_unc.shape == ((len(actions) - 1), n_agents)

            for lu in locs_unc:
                if np.all(lu == lu[0]): continue  # collision
                c_unc = _location_to_index(locations, lu)
                T[c_unc, i, c] += (1 - det) / (len(locs_unc))

    return T


def compute_transition_nstep(T, n_step):
    n_states, n_actions, _ = T.shape
    nstep_actions = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
    Bn = np.zeros([n_states, len(nstep_actions), n_states], dtype='uint8')
    for i, an in enumerate(nstep_actions):
        Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))
    return Bn