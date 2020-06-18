import numpy as np



def _index_to_cell(s, dims):
    cell = [int(s / dims[1]), s % dims[1]]
    return np.array(cell)


def _cell_to_index(cell, dims):
    return cell[1] + dims[1]*cell[0]


def _location_to_index(locations, locs):
    return np.where(np.all(locations == locs, axis=1))[0][0]


def _dist_locs(p1, p2, dims, round_off_func=lambda x: x):
    def _dir_to_dist(p_pos1, p_pos2):
        delta_pos = p_pos2 - p_pos1
        return round_off_func(np.sqrt(np.sum(np.square(delta_pos))))

    return _dir_to_dist(_index_to_cell(p1, dims), _index_to_cell(p2, dims))


vecmod = np.vectorize(lambda x, y : x % y)


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


