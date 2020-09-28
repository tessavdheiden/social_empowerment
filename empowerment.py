import numpy as np
import torch
from torch.autograd import Variable


from algorithms.maddpg import onehot_from_logits


class BaseEmpowerment(object):
    def __init__(self):
        pass

    def compute(self, reward, next_obs):
        return NotImplementedError

    def update(self, sample):
        pass

class DummyEmpowerment(BaseEmpowerment):
    def __init__(self, agents):
        super(DummyEmpowerment, self).__init__()
        self.agents = agents

    def compute(self, reward, next_obs):
        return reward


class JointEmpowerment(BaseEmpowerment):
    def __init__(self, agents):
        super(JointEmpowerment, self).__init__()
        self.agents = agents
        self.cell_num = 8
        self.cell_size = .5
        self.n_land = 3
        self.moves = np.array([[0, 0], [0, .1], [0, -.1],
                               [.1, 0], [-.1, 0],[0, .05],
                               [0, -.05], [.05, 0], [-.05, 0]])

        self.not_in_collision = lambda x: (len(x) == len(np.unique(x))) & np.all(x != 'inf') & np.all(x != '-inf')

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

    def compute(self, reward, next_obs):
        n_moves = len(self.moves)

        # get positions
        pos_obs_ag = next_obs[0, :, 2:2 + 2]

        # new positions
        batch_obs_ag = np.repeat(pos_obs_ag.reshape(1, self.n_land, 2), n_moves, axis=0) + np.repeat(
            self.moves.reshape(n_moves, 1, 2), self.n_land, axis=1)

        encoded = self.get_grid_indices(batch_obs_ag)
        unique_config = np.unique(encoded.reshape(n_moves, self.n_land), axis=0)
        new_unique_config = unique_config[np.apply_along_axis(self.not_in_collision, 1, unique_config)]
        return len(new_unique_config) * reward


class TransferEmpowerment(BaseEmpowerment):
    def __init__(self, agents):
        super(TransferEmpowerment, self).__init__()
        self.agents = agents
        self.n_channels = 3
        self.n_land = 3
        self.moves = np.array([[0, 0], [0, .1], [0, -.1],
                               [.1, 0], [-.1, 0],[0, .05],
                               [0, -.05], [.05, 0], [-.05, 0]])

        self.messages = np.zeros((self.n_channels + 1, self.n_channels))

        self.rep_rows = lambda x, n: np.repeat(np.expand_dims(x, 0), n, 0)
        self.rep_cols = lambda x, n: np.repeat(np.expand_dims(x, 1), n, 1)

        self.cell_num = 20
        self.cell_size = .1

    def compute(self, reward, next_obs):
        return reward * (self.empowerment_over_moves(next_obs) + self.empowerment_over_messages(next_obs))

    def empowerment_over_moves(self, next_obs):

        cast = lambda x: Variable(torch.Tensor(x), requires_grad=False)

        n_messages = len(self.messages)

        # filter out landmarks
        land_pos = next_obs[:, 1][0][2:2 + self.n_land * 2]

        # filter out messages
        pos_obs = next_obs[:, 1][0][:-self.n_channels]

        # create future configuratins
        batch_obs = np.repeat(pos_obs.reshape(1, -1), n_messages, axis=0)
        torch_obs = cast(np.concatenate((batch_obs, self.messages), axis=1))
        logits = self.agents[1].policy(torch_obs)
        actions = onehot_from_logits(logits)
        moves = self.moves[actions.max(1)[1]]

        next_land_pos = self.rep_rows(land_pos.reshape(-1, 2), n_messages) - self.rep_cols(moves, self.n_land)

        next_grid_indices = self.get_grid_indices(next_land_pos).reshape(n_messages, self.n_channels)
        assert next_grid_indices.shape[0] == next_land_pos.shape[0]

        # return unique set
        unique_config = np.unique(next_grid_indices, axis=0)
        assert unique_config.shape[1] == self.n_land

        return len(unique_config)

    def empowerment_over_messages(self, next_obs):
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
        logits = self.agents[0].policy(torch_obs)
        messages = onehot_from_logits(logits)

        encoded = messages.max(1)[1]
        unique_config = np.unique(encoded, axis=0)
        return len(unique_config)

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