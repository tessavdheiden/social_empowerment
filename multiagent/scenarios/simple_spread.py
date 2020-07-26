import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


colors = np.array([[0.65, 0.15, 0.15], [0.15, 0.65, 0.15], [0.15, 0.15, 0.65],
                   [0.15, 0.65, 0.65], [0.65, 0.15, 0.65], [0.65, 0.65, 0.15]])


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = colors[i]
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    collisions += 1
        return (self.reward(agent, world), collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def done(self, agent, world):
        pass


import torch
from torch.autograd import Variable
from multiagent.scenarios.mdp import BaseMDP
from multiagent.scenarios.transition_utils import switch_places, _location_to_index, _index_to_cell, _cell_to_index, vecmod
import itertools
from functools import reduce


class MDP(BaseMDP):
    def __init__(self, n_agents, dims, n_step):
        self.cell_num = 8
        self.cell_size = .5

        self.configurations = np.array(list(itertools.permutations(np.arange(dims[0] * dims[1]), n_agents)))
        self.actions = {
            "N": np.array([1, 0]),      # UP
            "S": np.array([-1, 0]),     # DOWN
            "E": np.array([0, 1]),      # RIGHT
            "W": np.array([0, -1]),     # LEFT
            "_": np.array([0, 0])       # STAY
        }
        self.moves = np.array([[0, 0], [0, .1], [0, -.1], [.1, 0], [-.1, 0],
                               [0, .05], [0, -.05], [.05, 0], [-.05, 0]])

        self.T = self.compute_transition(n_agents=n_agents, dims=dims, locations=self.configurations)
        #self.Tn = self.compute_transition_nstep(T=self.T, n_step=n_step)
        self.not_in_collision = lambda x: (len(x) == len(np.unique(x))) & np.all(x != 'inf') & np.all(x != '-inf')

    def act(self, s, a, dims, prob=1., toroidal=False):
        """ get updated state after action
        s  : state, index of grid position
        a : action
        prob : probability of performing action
        """
        rnd = np.random.rand()
        if rnd > prob:
            a = np.random.choice(list(filter(lambda x: x != a, self.actions.keys())))
        state = _index_to_cell(s, dims)

        new_state = state + self.actions[a]
        # can't move off grid
        if toroidal:
            new_state = vecmod(new_state, dims)
        elif np.any(new_state < np.zeros(2)) or np.any(new_state >= dims):
            return _cell_to_index(state, dims)
        return _cell_to_index(new_state, dims)


    def compute_transition(self, n_agents, dims, locations, det=1.):
        """ Computes probabilistic model T[s',a,s] corresponding to a grid world with N agents. """
        a_list = list(itertools.product(self.actions.keys(), repeat=n_agents))
        n_actions = len(a_list)

        n_configs = len(locations)
        # compute environment dynamics as a matrix T
        T = np.zeros([n_configs, n_actions, n_configs])
        # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
        for c, locs in enumerate(locations):
            for i, alist in enumerate(a_list):
                locs_new = [self.act(locs[j], alist[j], dims) for j in range(n_agents)]

                # any of the agents on same location, do not move
                if len(set(locs_new)) < n_agents or switch_places(locs, locs_new):
                    locs_new = locs

                c_new = _location_to_index(locations, locs_new)
                T[c_new, i, c] += det

                if det == 1: continue
                locs_unc = np.array(
                    [list(map(lambda x: self.act(locs[j], x, dims, det), filter(lambda x: x != alist[j], self.actions.keys())))
                     for j in range(n_agents)]).T
                assert locs_unc.shape == ((len(self.actions) - 1), n_agents)

                for lu in locs_unc:
                    if np.all(lu == lu[0]): continue  # collision
                    c_unc = _location_to_index(locations, lu)
                    T[c_unc, i, c] += (1 - det) / (len(locs_unc))
        return T

    def compute_transition_nstep(self, T, n_step):
        n_states, n_actions, _ = T.shape
        nstep_actions = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
        Bn = np.zeros([n_states, len(nstep_actions), n_states], dtype='uint8')
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))
        return Bn

    def get_unique_next_states(self, obs, next_obs, n_landmarks):
        cast = lambda x: Variable(torch.Tensor(x), requires_grad=False)

        n_moves = len(self.moves)

        # get positions
        pos_obs_ag = next_obs[0, :, 2:2+2]
        #pos_obs_target = next_obs[0][:][4:4+2*n_landmarks]

        # new positions
        batch_obs_ag = np.repeat(pos_obs_ag.reshape(1, n_landmarks, 2), n_moves, axis=0) + np.repeat(self.moves.reshape(n_moves, 1, 2), n_landmarks, axis=1)
        #batch_obs_target = np.repeat(pos_obs_target.reshape(1, n_landmarks, 2), n_moves, axis=0) + np.repeat(self.moves.reshape(n_moves, 1, 2), n_landmarks, axis=1)
        #batch_obs = np.concatenate((batch_obs_ag, batch_obs_target), axis=1)
        #torch_obs = cast(batch_obs)

        # messages
        #logits = self.actor.agents[0].policy(torch_obs)
        #messages = onehot_from_logits(logits)

        #encoded = messages.max(1)[1]
        encoded = self.get_grid_indices(batch_obs_ag)
        unique_config = np.unique(encoded.reshape(n_moves, n_landmarks), axis=0)
        new_unique_config = unique_config[np.apply_along_axis(self.not_in_collision, 1, unique_config)]
        # if len(unique_config)>len(new_unique_config):
        #     print(unique_config)
        #     print(new_unique_config)

        return len(new_unique_config) # TODO: return something which can be used for Blahut

    def get_grid_indices(self, obs):
        p_land = obs.reshape(-1, 2)
        other_px, other_py = p_land[:, 0], p_land[:, 1]
        other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
        other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
        other_x_index[other_x_index < 0] = float('-inf')
        other_x_index[other_x_index >= self.cell_num] = float('-inf')
        other_y_index[other_y_index < 0] = float('-inf')
        other_y_index[other_y_index >= self.cell_num] = float('-inf')
        grid_indices = self.cell_num * other_y_index + other_x_index
        return grid_indices