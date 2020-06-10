import numpy as np

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


