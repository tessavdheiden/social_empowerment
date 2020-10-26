import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
MSELoss = torch.nn.MSELoss()


from utils.misc import gumbel_softmax
from utils.networks import MLPNetwork
from empowerment import Device, BaseEmpowerment


class SocialInfluence(BaseEmpowerment):
    def __init__(self, agents, num_in_trans, num_out_trans, lr=0.01, hidden_dim=64, recurrent=False,
                 convolutional=False):
        super(SocialInfluence, self).__init__()
        self.agents = agents
        self.device = Device('cpu')
        self.transition = MLPNetwork(num_in_trans, num_out_trans, recurrent=True)

    def compute(self, reward, next_obs):
        return reward

    def update(self, sample, logger):
        pass

    def prep_training(self, device='gpu'):
        self.transition.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)

        self.device.set_device(device)

    def prep_rollouts(self, device='cpu'):
        self.transition.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)

        self.device.set_device(device)

    @classmethod
    def init(cls, agents, env, lr=0.01, hidden_dim=64, recurrent=False, convolutional=False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        init_params = []

        num_in_transition = num_out_transition = 0
        for i, (acsp, obsp) in enumerate(zip(env.action_space, env.observation_space)):

            num_in_transition += obsp.shape[0] + acsp.n
            num_out_transition += obsp.shape[0]


        init_dict = {'agents': agents,
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                     'num_in_trans': num_in_transition,
                     'num_out_trans': num_out_transition,
                     'recurrent': recurrent,
                     'convolutional': convolutional}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance