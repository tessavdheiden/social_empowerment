import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
MSELoss = torch.nn.MSELoss()


from utils.misc import gumbel_softmax
from utils.networks import MLPNetwork
from empowerment import Device, BaseEmpowerment


class Computer(object):
    def __init__(self, si):
        self.transition = si.transition
        self.planning = si.planning
        self.device = si.device
        self.agents = si.agents

    def compute(self, rewards, next_obs):
        with torch.no_grad():
            next_obs = [Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                      requires_grad=False) for i in range(rewards.shape[1])]

            acs_pi = []
            prob_pi = []
            for no, pi in zip(next_obs, self.agents):
                acs_pi.append(gumbel_softmax(pi.policy(no), device=self.device.get_device(), hard=True))
                prob_pi.append(gumbel_softmax(pi.policy(no), device=self.device.get_device(), hard=False))

            trans_in = torch.cat((*next_obs, *acs_pi), dim=1)
            trans_out = self.transition(trans_in)
            prob_plan = []

            end_idx = [0] + np.cumsum([ne_ob.shape[1] for ne_ob in next_obs]).tolist()
            start_end = [(start, end) for start, end in zip(end_idx, end_idx[1:])]
            for i, (no, planning) in enumerate(zip(next_obs, self.planning)):
                nno = trans_out[:, start_end[i][0]:start_end[i][1]]
                acs_ = []
                for j, ac in enumerate(acs_pi):
                    if j == i: continue # computing effect on other agent
                    nno_other = trans_out[:, start_end[j][0]:start_end[j][1]]
                    acs_.append(gumbel_softmax(self.agents[j].policy(nno_other), device=self.device.get_device(), hard=True))
                acs_ = torch.cat(acs_, dim=1)
                plan_in = torch.cat((no, nno, acs_), dim=1)

                prob_plan.append(gumbel_softmax(planning(plan_in), device=self.device.get_device(), hard=False))

            prob_plan = torch.cat(prob_plan, dim=1)
            prob_pi = torch.cat(prob_pi, dim=1)
            acs_pi = torch.cat(acs_pi, dim=1)

            SI = acs_pi * prob_plan - acs_pi * prob_pi
            i_rews = SI.mean() * torch.ones((1, rewards.shape[1]))
            return i_rews.numpy()


class Trainer(object):
    def __init__(self, si):
        self.transition = si.transition
        self.planning = si.planning
        self.device = si.device
        self.agents = si.agents

        self.transition_optimizer = Adam(self.transition.parameters(), lr=si.lr)
        flatten = lambda l: [item for sublist in l for item in sublist]
        params_planning = flatten([list((mlp.parameters())) for mlp in self.planning])
        params_agents = flatten([list(a.policy.parameters()) for a in self.agents])
        self.pi_optimizer = Adam(params_agents, lr=si.lr)
        self.niter = 0

    def update(self, sample, logger):

        obs, acs, rews, emps, next_obs, dones = sample

        self.transition_optimizer.zero_grad()
        trans_in = torch.cat((*obs, *acs), dim=1)
        next_obs_pred = self.transition(trans_in)
        trans_loss = MSELoss(next_obs_pred, torch.cat(next_obs, dim=1))
        trans_loss.backward()
        self.transition_optimizer.step()

        acs_pi = []
        prob_pi = []
        for no, pi in zip(next_obs, self.agents):
            acs_pi.append(gumbel_softmax(pi.policy(no), device=self.device.get_device(), hard=True))
            prob_pi.append(gumbel_softmax(pi.policy(no), device=self.device.get_device(), hard=False))
        with torch.no_grad():
            trans_in = torch.cat((*next_obs, *acs_pi), dim=1)
            trans_out = self.transition(trans_in)
        prob_plan = []

        end_idx = [0] + np.cumsum([ne_ob.shape[1] for ne_ob in next_obs]).tolist()
        start_end = [(start, end) for start, end in zip(end_idx, end_idx[1:])]
        for i, (no, planning) in enumerate(zip(next_obs, self.planning)):
            nno = trans_out[:, start_end[i][0]:start_end[i][1]]
            acs_ = []
            for j, ac in enumerate(acs):
                if j == i: continue
                nno_other = trans_out[:, start_end[j][0]:start_end[j][1]]
                acs_.append(gumbel_softmax(self.agents[j].policy(nno_other), device=self.device.get_device(), hard=True))
            acs_ = torch.cat(acs_, dim=1)
            plan_in = torch.cat((no, nno, acs_), dim=1)
            prob_plan.append(gumbel_softmax(planning(plan_in), device=self.device.get_device(), hard=False))

        prob_plan = torch.cat(prob_plan, dim=1)
        prob_pi = torch.cat(prob_pi, dim=1)
        acs_pi = torch.cat(acs_pi, dim=1)

        #self.pi_optimizer.zero_grad()
        SI = acs_pi * prob_plan - acs_pi * prob_pi
        i_rews = -SI.mean()
        #i_rews.backward()
        #self.pi_optimizer.step()

        if logger is not None:
            logger.add_scalars('si/losses',
                               {'trans_loss': trans_loss.detach(),
                                'i_rews': i_rews.detach()},
                               self.niter)
        self.niter += 1


class SocialInfluence(BaseEmpowerment):
    def __init__(self, agents, init_params, num_in_trans, num_out_trans, lr=0.01, hidden_dim=64, recurrent=False,
                 convolutional=False):
        super(SocialInfluence, self).__init__()
        self.agents = agents
        self.device = Device('cpu')
        self.transition = MLPNetwork(num_in_trans, num_out_trans, recurrent=True)
        self.planning = [MLPNetwork(p['num_in_plan'], p['num_out_plan'], recurrent=True) for p in init_params]

        self.lr = lr

        self.niter = 0

        self.computer = Computer(self)
        self.trainer = Trainer(self)

    def compute(self, rewards, next_obs):
        return self.computer.compute(rewards, next_obs)

    def update(self, sample, logger=None):
        return self.trainer.update(sample, logger)

    def prep_training(self, device='gpu'):
        self.transition.train()
        for planning in self.planning:
            planning.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)
            for planning in self.planning:
                planning = fn(planning)

        self.device.set_device(device)

    def prep_rollouts(self, device='cpu'):
        self.transition.eval()
        for planning in self.planning:
            planning.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)
            for planning in self.planning:
                planning = fn(planning)

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

            num_in_planning = 2 * obsp.shape[0]
            for j, acsp_j in enumerate(env.action_space):
                if j != i: num_in_planning += acsp_j.n
            num_out_planning = acsp.n
            init_params.append({'num_in_plan': num_in_planning,
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