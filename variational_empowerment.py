import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
MSELoss = torch.nn.MSELoss()


from utils.networks import MLPNetwork
from utils.misc import gumbel_softmax


class VariationalJointEmpowerment(object):
    def __init__(self, init_params, lr=0.01):
        super(VariationalJointEmpowerment, self).__init__()
        self.transition = MLPNetwork(init_params['num_in_trans'], init_params['num_out_trans'], recurrent=True)
        self.source = MLPNetwork(init_params['num_in_src'], init_params['num_out_src'], recurrent=True)
        self.planning = MLPNetwork(init_params['num_in_plan'], init_params['num_out_plan'], recurrent=True)

        self.transition_optimizer = Adam(self.transition.parameters(), lr=lr)
        self.source_optimizer = Adam(list(self.source.parameters()) + list(self.planning.parameters()), lr=lr)
        self.trans_dev = 'cpu'  # device for transition
        self.source_dev = 'cpu'
        self.plan_dev = 'cpu'

    def compute(self, rewards, obs):
        with torch.no_grad():
            obs_torch = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                      requires_grad=False) for i in range(rewards.shape[1])]
            obs_torch = torch.cat(obs_torch, dim=1)
            acs = self.source(obs_torch)
            next_obs_trans = self.transition(torch.cat((obs_torch, acs), dim=1))
            plan_in = torch.cat((obs_torch, next_obs_trans), dim=1)
            acs_plan = self.planning(plan_in)
            emp = gumbel_softmax(acs_plan, device=self.source_dev) - gumbel_softmax(acs, device=self.source_dev)
            i_rews = emp.mean() * torch.ones((1, rewards.shape[1]))
            return i_rews.numpy()

    def update(self, sample):
        obs, acs, rews, emps, next_obs, dones = sample

        self.transition_optimizer.zero_grad()
        trans_in = torch.cat((*obs, *acs), dim=1)
        next_obs_pred = self.transition(trans_in)
        trans_loss = MSELoss(next_obs_pred, torch.cat(next_obs, dim=1))
        trans_loss.backward()
        self.transition_optimizer.step()

        self.source_optimizer.zero_grad()
        obs_torch = torch.cat(next_obs, dim=1)

        acs_src = self.source(obs_torch)
        with torch.no_grad():
            trans_in = torch.cat((obs_torch, acs_src), dim=1)
            next_obs_trans = self.transition(trans_in)
        plan_in = torch.cat((obs_torch, next_obs_trans), dim=1)
        acs_plan = self.planning(plan_in)

        emp = - gumbel_softmax(acs_plan, device=self.source_dev) + gumbel_softmax(acs_src, device=self.source_dev)
        emp.mean().backward()
        self.source_optimizer.step()

    def prep_training(self, device='gpu'):
        self.transition.train()
        self.source.train()
        self.planning.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.trans_dev == device:
            self.transition = fn(self.transition)
            self.trans_dev = device
        if not self.source_dev == device:
            self.source = fn(self.source)
            self.source_dev = device
        if not self.plan_dev == device:
            self.planning = fn(self.planning)
            self.plan_dev = device

    def prep_rollouts(self, device='cpu'):
        self.transition.eval()
        self.source.eval()
        self.planning.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.trans_dev == device:
            self.transition = fn(self.transition)
            self.trans_dev = device
        if not self.source_dev == device:
            self.source = fn(self.source)
            self.source_dev = device
        if not self.plan_dev == device:
            self.planning = fn(self.planning)
            self.plan_dev = device

    @classmethod
    def init_from_env(cls, env):
        num_in_source = num_out_source = num_in_planning = \
            num_out_planning = num_in_transition = num_out_transition = 0
        for acsp, obsp in zip(env.action_space, env.observation_space):

            num_in_source += obsp.shape[0]
            num_out_source += acsp.n

            num_in_planning += 2*obsp.shape[0]
            num_out_planning += acsp.n

            num_in_transition += obsp.shape[0] + acsp.n
            num_out_transition += obsp.shape[0]

        init_params = {'num_in_src': num_in_source,
                        'num_in_plan': num_in_planning,
                        'num_in_trans': num_in_transition,
                        'num_out_src': num_out_source,
                        'num_out_plan': num_out_planning,
                        'num_out_trans': num_out_transition}

        instance = cls(init_params)
        instance.init_dict = init_params
        return instance



