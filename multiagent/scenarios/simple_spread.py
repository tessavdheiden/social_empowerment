import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


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
            agent.color = np.array([0.35, 0.35, 0.85])
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
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


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


from multiagent.scenarios.mdp import BaseMDP
from multiagent.scenarios.transition_utils import switch_places, _location_to_index, _index_to_cell, _cell_to_index, vecmod
import itertools
from functools import reduce


class MDP(BaseMDP):
    def __init__(self, n_agents, dims, n_step):

        self.configurations = np.array(list(itertools.permutations(np.arange(dims[0] * dims[1]), n_agents)))
        self.actions = {
            "N": np.array([1, 0]),      # UP
            "S": np.array([-1, 0]),     # DOWN
            "E": np.array([0, 1]),      # RIGHT
            "W": np.array([0, -1]),     # LEFT
            "_": np.array([0, 0])       # STAY
        }

        self.T = self.compute_transition(n_agents=n_agents, dims=dims, locations=self.configurations)
        self.Tn = self.compute_transition_nstep(T=self.T, n_step=n_step)

    def act(self, s, a, dims, prob=1., toroidal=False):
        """ get updated state after action
        s  : state, index of grid position
        a : action
        prob : probability of performing action
        """
        rnd = np.random.rand()
        if rnd > prob:
            a = np.random.choice(list(filter(lambda x: x != a, actions.keys())))
        state = _index_to_cell(s, dims)

        new_state = state + self.actions[a]
        # can't move off grid
        if toroidal:
            new_state = vecmod(new_state, dims)
        elif np.any(new_state < np.zeros(2)) or np.any(new_state >= dims):
            return _cell_to_index(state, dims)
        return _cell_to_index(new_state, dims)


    def compute_transition(self, n_agents, dims, locations, det=1.):

        a_list = list(itertools.product(self.actions.keys(), repeat=n_agents))
        n_actions = len(a_list)

        n_configs = len(locations)
        # compute environment dynamics as a matrix T
        T = np.zeros([n_configs, n_actions, n_configs], dtype='uint8')
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