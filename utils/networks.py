import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, recurrent=False,
                 convolutional=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        if convolutional:
            self.fc1 = ConvolutionalUnit(input_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)

        if recurrent:
            self.fc2 = RecurrentUnit(hidden_dim, hidden_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class RecurrentUnit(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(RecurrentUnit, self).__init__()
        self.lstm = nn.LSTM(input_dim, out_dim)
        self.h_0 = nn.Parameter(torch.randn(input_dim))
        self.c_0 = nn.Parameter(torch.randn(input_dim))

    def init_state(self, batch_size):
        h_0 = self.h_0.repeat(1, batch_size, 1)
        c_0 = self.c_0.repeat(1, batch_size, 1)
        return (h_0, c_0)

    def forward(self, x):
        batch_size, feat_size = x.shape
        state = self.init_state(batch_size)
        x, _ = self.lstm(x.unsqueeze(0), state)
        return x.view(batch_size, feat_size)


class ConvolutionalUnit(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(ConvolutionalUnit, self).__init__()
        self.cnn = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(1, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        ) # output shape (256, 1, 1)
        self.fc = nn.Sequential(nn.Linear(256, out_dim), nn.ReLU())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        batch_size, feat_dim = x.shape
        w = int(feat_dim ** .5)
        x = self.cnn(x.view(batch_size, 1, w, w))
        x = x.view(-1, 256)
        return self.fc(x)