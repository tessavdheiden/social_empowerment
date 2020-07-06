import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return (self.reward(agent, world), )

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)


from multiagent.scenarios.mdp import BaseMDP
from estimate_empowerment import empowerment
import itertools
from functools import reduce
from utils.agents import onehot_from_logits
import torch
from torch.autograd import Variable
import sys
np.set_printoptions(threshold=sys.maxsize)

rep_rows = lambda x, n: np.repeat(np.expand_dims(x, 0), n, 0)
rep_cols = lambda x, n: np.repeat(np.expand_dims(x, 1), n, 1)


class MDP(BaseMDP):
    def __init__(self, actor, n_landmarks, n_channels):
        self.actor = actor
        self.actor.prep_rollouts(device='cpu')
        self.n_lm = n_landmarks
        self.n_ch = n_channels
        self.dim = 1

        self.cell_num = 20
        self.cell_size = .1

        self.roff = lambda x: np.around(x / self.cell_size) * self.cell_size # TODO: reconsider as function of dim

        #self.sspa = self._make_sspa(n_landmarks)

        # listener's actions
        self.moves = np.array([[0, 0], [0, .1], [0, -.1], [.1, 0], [-.1, 0],
                               [0, .05], [0, -.05], [.05, 0], [-.05, 0]]) # STOP, UP, DOWN, RIGHT, LEFT # TODO: reconsider delta x

        self.messages = np.zeros((n_channels + 1, n_channels))
        for i in range(1, n_channels + 1):
            self.messages[i, i - 1] = 1

        # transition function
        self.T = None
        # experience
        self.D = None


    def get_unique_next_states(self, obs, next_obs, n_landmarks):

        cast = lambda x: Variable(torch.Tensor(x), requires_grad=False)

        n_messages = len(self.messages)

        # filter out landmarks
        land_pos = next_obs[:, 1][0][2:2 + n_landmarks * 2]

        # filter out messages
        pos_obs = next_obs[:, 1][0][:-self.n_ch]

        # create unique configuration of current obs
        # grid_indices = self.get_grid_indices(land_pos)
        # occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)

        # create future configuratins
        batch_obs = np.repeat(pos_obs.reshape(1, -1), n_messages, axis=0)
        torch_obs = cast(np.concatenate((batch_obs, self.messages), axis=1))
        logits = self.actor.agents[1].policy(torch_obs)
        actions = onehot_from_logits(logits)
        moves = self.moves[actions.max(1)[1]]

        next_land_pos = rep_rows(land_pos.reshape(-1, 2), n_messages) - rep_cols(moves, n_landmarks)

        next_grid_indices = self.get_grid_indices(next_land_pos).reshape(n_messages, n_landmarks)
        assert next_grid_indices.shape[0] == next_land_pos.shape[0]

        # return unique set
        unique_config = np.unique(next_grid_indices, axis=0)
        assert unique_config.shape[1] == n_landmarks

        return len(unique_config) # TODO: return something which can be used for Blahut

    def get_unique_next_states_speaker(self, obs, next_obs, n_landmarks):
        cast = lambda x: Variable(torch.Tensor(x), requires_grad=False)

        n_moves = len(self.moves)

        # get positions
        pos_obs_ag = next_obs[:, 0][0][0:2]
        pos_obs_target = next_obs[:, 0][0][2:]

        # new positions
        batch_obs_ag = np.repeat(pos_obs_ag.reshape(1, -1), n_moves, axis=0) + self.moves
        batch_obs_target = np.repeat(pos_obs_target.reshape(1, -1), n_moves, axis=0)
        batch_obs = np.concatenate((batch_obs_ag, batch_obs_target), axis=1)
        torch_obs = cast(batch_obs)

        # messages
        logits = self.actor.agents[0].policy(torch_obs)
        messages = onehot_from_logits(logits)

        encoded = messages.max(1)[1]
        unique_config = np.unique(encoded, axis=0)
        return len(unique_config) # TODO: return something which can be used for Blahut

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


def rand_sample(p_x):
    """
    Randomly sample a value from a probability distribution p_x
    """
    cumsum = np.cumsum(p_x)
    rnd = np.random.rand()
    return np.argmax(cumsum > rnd)


def normalize(X):
    """
    Normalize vector or matrix columns X
    """
    return X / X.sum(axis=0)


def softmax(x, tau):
    """
    Returns the softmax normalization of a vector x using temperature tau.
    """
    return normalize(np.exp(x / tau))


def load_agent():
    from algorithms.maddpg import MADDPG
    maddpg = MADDPG.init_from_save('models/simple_speaker_listener/my_model/run1/model.pt')
    maddpg.prep_rollouts(device='cpu')
    return maddpg


def draw_config(l_s, row, col, ax):
    ax[row, col].cla()
    from matplotlib.patches import Circle
    colors = ['r', 'g', 'b']
    for cell, c in zip(l_s, colors):
        ax[row, col].add_patch(Circle((cell[0], cell[1]), .5, color=c))

    ax[row, col].set_ylim(-1.5, 1.5)
    ax[row, col].set_xlim(-1.5, 1.5)



