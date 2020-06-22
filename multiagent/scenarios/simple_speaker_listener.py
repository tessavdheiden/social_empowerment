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
from multiagent.scenarios.transition_utils import _dist_locs, _index_to_cell, _cell_to_index, vecmod
from estimate_empowerment import empowerment
import itertools
from functools import reduce
from utils.agents import onehot_from_logits
import torch
from torch.autograd import Variable
cast = lambda x: Variable(torch.Tensor(x).view(1, -1), requires_grad=False)
roff = lambda x: np.around(x, decimals=0)

class MDP(BaseMDP):
    def __init__(self, n_step, agent):
        self.agent = agent
        self.agent.prep_rollouts(device='cpu')

        self.sspa = gen_state_space()
        self.aspa = gen_action_space()

        # transition function
        self.T = self.compute_transition(self.sspa, self.aspa, self.agent)
        self.Tn = self.compute_transition_nstep(T=self.T, n_step=n_step)
        # experience
        self.D = self.T.copy()

    def act(self, s, a, agent, cast):
        dr, dg, db = s
        obs = cast(np.concatenate((np.array([0, 0]), dr, dg, db, a), axis=0))
        return agent.agents[1].policy(obs)

    def compute_transition(self, sspa, aspa, agent):

        T = np.zeros((len(sspa), len(sspa), len(sspa)))
        for s, config in enumerate(sspa):
            for i, comm in enumerate(aspa):
                a = self.act(config, comm, agent, cast)
                a = onehot_from_logits(a)
                move = move_from_onehot(a)

                config_ = propagate_state(config, move)
                s_ = find_state(config_, sspa)

                T[s_, i, s] += 1

        return T

    def update_transition(self, config, comm, agent):
        s = find_state(config, self.sspa)
        logits = self.act(config, comm, agent, cast)
        action = onehot_from_logits(logits)
        move = move_from_onehot(action)

        config_ = propagate_state(config, move)
        s_ = find_state(config_, self.sspa)
        a = np.where(np.all(self.aspa == comm, 1))[0]
        self.D[s_, a, s] += 1
        self.T[:, move, s] = normalize(self.D[:, a, s])


    def compute_transition_nstep(self, T, n_step):
        n_states, n_actions, _ = T.shape
        nstep_actions = np.array(list(itertools.product(range(n_actions), repeat=n_step)))
        Bn = np.zeros([n_states, len(nstep_actions), n_states], dtype='uint8')
        for i, an in enumerate(nstep_actions):
            Bn[:, i, :] = reduce((lambda x, y: np.dot(y, x)), map((lambda a: T[:, a, :]), an))
        return Bn

    def config_from_pos(self, p_land, p_agent):
        return roff(p_land - p_agent)


def plot_config(l_s, a, row, col, ax):
    for s in l_s:
        cell = _index_to_cell(s, dims)
        ax[row, col].add_patch(Circle((cell[1] + .5, cell[0] + .5), .25))
    cell = _index_to_cell(a, dims)
    ax[row, col].add_patch(Circle((cell[1] + .5, cell[0] + .5), .5, color='r', alpha=.5))
    ax[row, col].set_ylim(0, dims[0])
    ax[row, col].set_xlim(0, dims[1])


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

def gen_state_space():
    locations = [[0, 1],     # UP
                [0, -1],     # DOWN
                [1, 0],      # RIGHT
                [-1, 0],     # LEFT
                [0, 0],      # CENTER
                [1, 1],      # RIGHT UP
                [1, -1],     # RIGHT DOWN
                [-1, 1],     # LEFT UP
                [-1, -1],    # LEFT DOWN
                [2, 2]]
    a = np.array(list(itertools.permutations(locations, 3)))
    b = np.array([[i, j,[2, 2]] for i in locations for j in locations]) # ALL, 1 or 2 OUT
    result = np.concatenate((a, b.reshape(100, 3, 2)), axis=0)
    return result

def gen_action_space():
    return [np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])]

def move_from_onehot(action):
    moves = {
        "N": np.array([0, 1]),      # UP
        "S": np.array([0, -1]),     # DOWN
        "E": np.array([1, 0]),      # RIGHT
        "W": np.array([-1, 0]),     # LEFT
        "_": np.array([0, 0])       # STAY
    }

    action = np.argmax(action)
    if action == 1: return moves["N"]
    if action == 2: return moves["S"]
    if action == 3: return moves["E"]
    if action == 4: return moves["W"]
    if action == 0: return moves["_"]

def propagate_state(s, move):
    new_state = s - move
    for i, s_ in enumerate(new_state):
        if any(s_ > 1) or any(s_<-1):
            new_state[i] = np.array([2, 2])

    return new_state

def find_state(state, sspa):
    a = state.flatten()
    other = sspa.reshape(-1, 6)
    for i, s in enumerate(other):
        if all(s == a):
            return i
    return len(sspa) - 1

def draw_config(l_s, row, col, ax):
    ax[row, col].cla()
    from matplotlib.patches import Circle
    colors = ['r', 'g', 'b']
    for cell, c in zip(l_s, colors):
        ax[row, col].add_patch(Circle((cell[0], cell[1]), .5, color=c))

    ax[row, col].set_ylim(-1.5, 1.5)
    ax[row, col].set_xlim(-1.5, 1.5)



if __name__ == '__main__':



    import matplotlib.pyplot as plt



    fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(12, 6))

    agent = load_agent()
    sspa = gen_state_space()
    aspa = gen_action_space()

    T = np.zeros((len(sspa), len(aspa), len(sspa)))
    for s, config in enumerate(sspa):
        for i, comm in enumerate(aspa):
            a = act(config, comm, agent, cast)
            a = onehot_from_logits(a)
            move = move_from_onehot(a)

            config_ = propagate_state(config, move)
            s_ = find_state(config_, sspa)

            T[s_, i, s] += 1

    E = np.zeros(len(sspa))

    for s, state in enumerate(sspa):
        print(f'progress = {s} our of {len(sspa)}')
        E[s] = empowerment(T=T, det=1, n_step=1, state=s)
    idx = np.argsort(E)
    low_idx = np.where(E == E[idx[0]])[0]
    high_idx = np.where(E == E[idx[-1]])[0]

    count = 0
    while count < 3:
        idx = np.random.choice(low_idx)
        low_e_config = sspa[idx]
        draw_config(low_e_config, count, 0, ax)
        ax[count, 0].set_xlabel(f'E={E[idx]}')


        state = low_e_config
        for i, action in enumerate(aspa):
            new_action = act(state, action, agent, cast)
            new_action = onehot_from_logits(new_action)
            new_move = move_from_onehot(new_action)

            new_state = propagate_state(state, new_move)
            s_ = find_state(new_state, sspa)

            draw_config(new_state, count, i+1, ax)
            ax[count, i+1].set_xlabel(f'{action}, {s_}, {new_move}')
        count += 1

    while count < 6:
        idx = np.random.choice(high_idx)
        high_e_config = sspa[idx]
        draw_config(high_e_config, count, 0, ax)
        ax[count, 0].set_xlabel(f'E={E[idx]}')

        state = high_e_config
        for i, action in enumerate(aspa):
            new_action = act(state, action, agent, cast)
            new_action = onehot_from_logits(new_action)
            new_move = move_from_onehot(new_action)

            new_state = propagate_state(state, new_move)
            draw_config(new_state, count, i+1, ax)
            s_ = find_state(new_state, sspa)
            ax[count, i+1].set_xlabel(f'{action}, {s_}, {new_move}')
        count += 1


    plt.show()


    # plt.savefig('tmp.png')
