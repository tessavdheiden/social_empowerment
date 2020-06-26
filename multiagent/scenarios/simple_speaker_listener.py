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

SIZE = 3

cast = lambda x: Variable(torch.Tensor(x).view(1, -1), requires_grad=False)
if SIZE == 5:
    roff = lambda x: np.around(x *  2, decimals=0) / 2
else:
    roff = lambda x: np.around(x, decimals=0) # TODO: reconsider as function of dim

rep_rows = lambda x, n: np.repeat(np.expand_dims(x, 0), n, 0)
rep_cols = lambda x, n: np.repeat(np.expand_dims(x, 1), n, 1)


class MDP(BaseMDP):
    def __init__(self, actor, n_landmarks, n_channels):
        self.actor = actor
        self.actor.prep_rollouts(device='cpu')
        self.n_lm = n_landmarks
        self.n_ch = n_channels
        self.dim = 1
        self.size = SIZE

        self.sspa = self._make_sspa(n_landmarks)

        # listener's actions
        self.moves = {
            "_": np.array([0, 0]),      # STAY
            "N": np.array([0, 1]),      # UP
            "S": np.array([0, -1]),     # DOWN
            "E": np.array([1, 0]),      # RIGHT
            "W": np.array([-1, 0])      # LEFT
        } # TODO: reconsider delta x

        self.messages = np.zeros((n_channels + 1, n_channels))
        for i in range(1, n_channels + 1):
            self.messages[i, i - 1] = 1

        # transition function
        self.T = None
        # experience
        self.D = None

    def _make_sspa(self, n_landmarks):
        cells = np.linspace(-self.dim, self.dim, self.size)
        locations = list(itertools.product(cells, repeat=2))

        for _ in range(n_landmarks):
            locations.append((self.dim+1, self.dim+1))
        permutations = np.array(list(itertools.product(locations, repeat=n_landmarks)))
        #permutations = np.array(list(itertools.permutations(locations, n_landmarks)))
        unique_perm = np.unique(permutations.reshape(-1, n_landmarks*2), axis=0)

        return roff(unique_perm.reshape(-1, n_landmarks, 2))

    def _idx_s_not_in_bounds(self, s):
        return np.where(np.any(s > self.dim, 1) | np.any(s < -self.dim, 1))[0]

    def propagate_delta_pos(self, s, move):
        new_state = roff(s - move)
        new_state[self._idx_s_not_in_bounds(new_state), :] = np.array([self.dim+1, self.dim+1])
        return new_state

    def _find_idx_from_delta_pos(self, c, sspa):
        other = sspa.reshape(-1, self.n_lm * 2)
        return np.argmax(np.all(other == c.flatten(), 1))

    def _delta_landmark_pos(self, p_land, p_agent):
        delta_pos = p_land - p_agent
        delta_pos = roff(delta_pos)
        delta_pos[self._idx_s_not_in_bounds(delta_pos), :] = np.array([self.dim+1, self.dim+1])
        return delta_pos

    def act(self, s, a):
        """ get updated state after action
        s  : listener's relative positions to landmarks
        a : message of speaker
        prob : probability of performing action
        """
        obs = cast(np.concatenate((np.array([0, 0]), *s, a), axis=0))
        return self.actor.agents[1].policy(obs)

    def get_transition_for_state(self, p_land, p_agent):
        delta_pos = np.squeeze(self._delta_landmark_pos(p_land, p_agent))
        """ Get probabilistic model T[s',a] corresponding to a grid world with 2 agents n landmarks. """
        # transition function
        T = np.zeros((len(self.sspa), len(self.messages)))
        # experience
        D = T.copy()
        for i, comm in enumerate(self.messages):
            logits = self.act(delta_pos, comm)
            action = onehot_from_logits(logits)

            move = list(self.moves.values())[np.argmax(action)]

            delta_pos_ = self.propagate_delta_pos(delta_pos, move)
            s_ = self._find_idx_from_delta_pos(delta_pos_, self.sspa)
            a = np.where(np.all(self.messages == comm, 1))[0]
            assert i == a
            D[s_, a] += 1
            T[:, a] = normalize(D[:, a])
        return T

    def get_transition_for_state_batch_implementation(self, p_land, p_agent):
        to_torch = lambda x: Variable(torch.Tensor(x), requires_grad=False)
        delta_pos = np.squeeze(self._delta_landmark_pos(p_land, p_agent))
        """ Get probabilistic model T[s',a] corresponding to a grid world with 2 agents n landmarks. """
        # transition function
        T = np.zeros((len(self.sspa), len(self.messages)))
        # experience
        D = T.copy()
        n = len(self.messages)
        obs = np.repeat(np.concatenate((np.array([0, 0]), *delta_pos), axis=0).reshape(1, -1), n, axis=0)
        obs = to_torch(np.concatenate((obs, self.messages), axis=1))
        logits = self.actor.agents[1].policy(obs)
        actions = onehot_from_logits(logits)
        moves = np.array(list(self.moves.values()))[actions.max(1)[1]]

        new_states = roff(rep_rows(delta_pos, n) - rep_cols(moves, self.n_lm))

        new_states = new_states.reshape(-1, 2)
        new_states[self._idx_s_not_in_bounds(new_states), :] = np.array([self.dim+1, self.dim+1])
        new_states = new_states.reshape(n, self.n_lm, 2).reshape(n, self.n_lm * 2)
        idx = list(map(lambda x: np.where(np.all(self.sspa.reshape(-1, self.n_lm * 2) == x, 1))[0], new_states)) # TODO: Can idx be empty?
        #idx = list(map(lambda x: -1 if len(x) == 0 else x[0], idx))
        s_ = np.array(idx).reshape(-1)
        a = np.arange(len(self.messages))

        D[s_, a] += 1
        T[:, a] = normalize(D[:, a])

        return T

    def get_idx_from_positions(self, p_land, p_agent):
        delta_p = np.squeeze(self._delta_landmark_pos(p_land, p_agent))
        return self._find_idx_from_delta_pos(delta_p, self.sspa)


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



