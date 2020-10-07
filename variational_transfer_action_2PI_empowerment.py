import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
MSELoss = torch.nn.MSELoss()


from variational_empowerment import VariationalBaseEmpowerment
from utils.misc import gumbel_softmax
from utils.networks import MLPNetwork


class ComputerTransferActionPi(object):
    def __init__(self, empowerment):
        self.transition = empowerment.transition
        self.source = empowerment.source
        self.planning = empowerment.planning
        self.plan_dev = empowerment.plan_dev
        self.source_dev = empowerment.source_dev
        self.trans_dev = empowerment.trans_dev
        self.agents = empowerment.agents

    def compute(self, rewards, next_obs):
        with torch.no_grad():
            next_obs = [Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                      requires_grad=False) for i in range(rewards.shape[1])]

            acs_src = []
            prob_src = []
            for no, source in zip(next_obs, self.source):
                acs_src.append(gumbel_softmax(source(no), device=self.source_dev, hard=True))
                prob_src.append(gumbel_softmax(source(no), device=self.source_dev, hard=False))

            trans_in = torch.cat((*next_obs, *acs_src), dim=1)
            trans_out = self.transition(trans_in)
            prob_plan = []
            start = 0
            for i, (no, planning) in enumerate(zip(next_obs, self.planning)):
                length = no.shape[1]
                nno = trans_out[:, start:start + length]
                acs_ = []
                for j, ac in enumerate(acs_src):
                    if j == i: continue
                    acs_.append(gumbel_softmax(self.agents[j].policy(nno), device=self.source_dev, hard=True))
                acs_ = torch.cat(acs_, dim=1)
                plan_in = torch.cat((no, nno, acs_), dim=1)

                prob_plan.append(gumbel_softmax(planning(plan_in), device=self.plan_dev, hard=False))
            prob_plan = torch.cat(prob_plan, dim=1)
            prob_src = torch.cat(prob_src, dim=1)
            acs_src = torch.cat(acs_src, dim=1)

            E = acs_src * prob_plan - acs_src * prob_src
            i_rews = E.mean() * torch.ones((1, rewards.shape[1]))
            return i_rews.numpy()

    def prep_rollouts(self, device='cpu'):
        self.transition.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.trans_dev == device:
            self.transition = fn(self.transition)
            self.trans_dev = device
        self.transition.eval()
        if not self.source_dev == device:
            for source in self.source:
                source = fn(source)
            self.source_dev = device
        for source in self.source:
            source.eval()
        if not self.plan_dev == device:
            for planning in self.planning:
                planning = fn(planning)
            self.plan_dev = device
        for planning in self.planning:
            planning.eval()

    def prepare_training(self, device):
        self.trans_dev = device
        self.source_dev = device
        self.plan_dev = device


class TrainerTransferActionPi(object):
    def __init__(self, empowerment):
        self.transition = empowerment.transition
        self.source = empowerment.source
        self.planning = empowerment.planning
        self.plan_dev = empowerment.plan_dev
        self.source_dev = empowerment.source_dev
        self.trans_dev = empowerment.trans_dev
        self.agents = empowerment.agents

        self.transition_optimizer = Adam(self.transition.parameters(), lr=empowerment.lr)
        params_planning = []
        for mlp in self.planning: params_planning += list(mlp.parameters())
        self.planning_optimizer = Adam(params_planning, lr=empowerment.lr)
        params_source = []
        for mlp in self.source: params_source += list(mlp.parameters())
        params_agents = []
        for a in self.agents:
            params_agents += list(a.policy.parameters()) + list(a.critic.parameters())

        self.source_optimizer = Adam(params_source + params_planning + params_agents, lr=empowerment.lr)
        self.niter = 0

    def update(self, sample, logger):
        obs, acs, rews, emps, next_obs, dones = sample

        self.transition_optimizer.zero_grad()
        trans_in = torch.cat((*obs, *acs), dim=1)
        next_obs_pred = self.transition(trans_in)
        trans_loss = MSELoss(next_obs_pred, torch.cat(next_obs, dim=1))
        trans_loss.backward()
        self.transition_optimizer.step()

        self.source_optimizer.zero_grad()
        acs_src = []
        prob_src = []
        for no, source in zip(next_obs, self.source):
            acs_src.append(gumbel_softmax(source(no), device=self.source_dev, hard=True))
            prob_src.append(gumbel_softmax(source(no), device=self.source_dev, hard=False))
        with torch.no_grad():
            trans_in = torch.cat((*next_obs, *acs_src), dim=1)
            trans_out = self.transition(trans_in)
        prob_plan = []
        start = 0
        for i, (no, planning) in enumerate(zip(next_obs, self.planning)):
            length = no.shape[1]
            nno = trans_out[:, start:start + length]
            acs_ = []
            for j, ac in enumerate(acs):
                if j == i: continue
                acs_.append(gumbel_softmax(self.agents[j].policy(nno), device=self.source_dev, hard=True))
            acs_ = torch.cat(acs_, dim=1)
            plan_in = torch.cat((no, nno, acs_), dim=1)
            prob_plan.append(gumbel_softmax(planning(plan_in), device=self.plan_dev, hard=False))
            start += length
        prob_plan = torch.cat(prob_plan, dim=1)
        prob_src = torch.cat(prob_src, dim=1)
        acs_src = torch.cat(acs_src, dim=1)

        E = acs_src * prob_plan - acs_src * prob_src
        i_rews = -E.mean()
        i_rews.backward()
        self.source_optimizer.step()

        if logger is not None:
            logger.add_scalars('empowerment/losses',
                               {'trans_loss': trans_loss.detach(),
                                'i_rews': i_rews.detach()},
                               self.niter)
        self.niter += 1

    def prepare_training(self, device):
        self.transition.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.trans_dev == device:
            self.transition = fn(self.transition)
            self.trans_dev = device
        if not self.source_dev == device:
            for source in self.source:
                source = fn(source)
                source.train()
            self.source_dev = device
        if not self.plan_dev == device:
            for planning in self.planning:
                planning = fn(planning)
                planning.train()
            self.plan_dev = device

    def prep_rollouts(self, device='cpu'):
        self.trans_dev = device
        self.source_dev = device
        self.plan_dev = device


class VariationalTransferActionPiEmpowerment(VariationalBaseEmpowerment):
    def __init__(self, agents, init_params, num_in_trans, num_out_trans, lr=0.01, hidden_dim=64, recurrent=False,
                 convolutional=False):
        super(VariationalTransferActionPiEmpowerment, self).__init__()
        self.agents = agents
        self.transition = MLPNetwork(num_in_trans, num_out_trans, recurrent=True)
        self.source = [MLPNetwork(p['num_in_src'], p['num_out_src'], recurrent=True) for p in init_params]
        self.planning = [MLPNetwork(p['num_in_plan'], p['num_out_plan'], recurrent=True) for p in init_params]

        self.lr = lr

        self.trans_dev = 'cpu'  # device for transition
        self.source_dev = 'cpu'
        self.plan_dev = 'cpu'

        self.computer = ComputerTransferActionPi(self)
        self.trainer = TrainerTransferActionPi(self)

    def compute(self, rewards, next_obs):
        return self.computer.compute(rewards, next_obs)

    def update(self, sample, logger=None):
        return self.trainer.update(sample, logger)

    def prep_training(self, device='gpu'):
        self.computer.prepare_training(device)
        self.trainer.prepare_training(device)
        self.transition.train()

    def prep_rollouts(self, device='cpu'):
        self.computer.prep_rollouts()
        self.trainer.prep_rollouts()

    @classmethod
    def init(cls, agents, env, lr=0.01, hidden_dim=64, recurrent=False, convolutional=False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        init_params = []

        num_in_transition = num_out_transition = 0
        for i, (acsp, obsp) in enumerate(zip(env.action_space, env.observation_space)):
            num_in_source = obsp.shape[0]
            num_out_source = acsp.n

            num_in_planning = 2 * obsp.shape[0]
            for j, acsp_j in enumerate(env.action_space):
                if j != i: num_in_planning += acsp_j.n
            num_out_planning = acsp.n

            num_in_transition += obsp.shape[0] + acsp.n
            num_out_transition += obsp.shape[0]

            init_params.append({'num_in_src': num_in_source,
                    'num_in_plan': num_in_planning,
                    'num_out_src': num_out_source,
                    'num_out_plan': num_out_planning})

        init_dict = {'agents': agents,
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'init_params': init_params,
                     'num_in_trans': num_in_transition,
                     'num_out_trans': num_out_transition,
                     'recurrent': recurrent,
                     'convolutional': convolutional}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance