import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, recurrent=False, convolutional=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if convolutional:
            self.emb = ConvolutionalUnit(input_dim=4)
            input_dim = hidden_dim
        else:
            self.emb = lambda x: x

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

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
        X = self.emb(X)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class DMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_a_dim, input_x_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, recurrent=False, convolutional=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DMLPNetwork, self).__init__()

        if convolutional:
            self.embS = ConvolutionalUnit(input_dim=input_x_dim)
            input_dim = hidden_dim + input_a_dim
        else:
            self.embS = nn.Linear(input_x_dim, hidden_dim)
            input_dim = hidden_dim + input_a_dim

        self.embA = nn.Linear(input_a_dim, input_a_dim)

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

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

    def forward(self, I):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        (S, A) = (I[0], I[1])
        S = self.embS(S)
        A = self.embA(A)
        X = torch.cat((S, A), dim=1)
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
    def __init__(self, input_dim):
        super(ConvolutionalUnit, self).__init__()
        self.hidden_dim = 64
        self.out_dim = 64
        self.input_dim = input_dim
        #layer_helper = lambda i, k, s: (i - k) / s + 1
        self.cnn = nn.Sequential(  # input shape (4, 16, 16)
            nn.Conv2d(self.input_dim, 8, kernel_size=4, stride=2),       # out  (8, 7, 7)
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=1),      # out  (16, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=1),     # out (16, 3, 3)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=1),     # out (64, 1, 1)
            nn.ReLU(),  # activation
        )
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.out_dim), nn.ReLU())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        batch, channels, w, h = x.shape
        x = self.cnn(x.view(batch, channels, w, h))
        x = x.view(-1, self.hidden_dim)
        return self.fc(x)