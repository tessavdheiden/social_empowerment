import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
MSELoss = torch.nn.MSELoss()


from empowerment import Device
from variational_empowerment import VariationalBaseEmpowerment
from utils.misc import gumbel_softmax
from utils.networks import MLPNetwork


class ComputerJoint(object):
    def __init__(self, variational_joint_empowerment):
        self.transition = variational_joint_empowerment.transition
        self.source = variational_joint_empowerment.source
        self.planning = variational_joint_empowerment.planning

        self.device = variational_joint_empowerment.device

    def compute(self, rewards, next_obs):
        with torch.no_grad():
            next_obs = [Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                      requires_grad=False) for i in range(rewards.shape[1])]

            acs_src = []
            prob_src = []
            for no in next_obs:
                acs_src.append(gumbel_softmax(self.source(no), device=self.device.get_device(), hard=True))
                prob_src.append(gumbel_softmax(self.source(no), device=self.device.get_device(), hard=False))

            trans_in = torch.cat((*next_obs, *acs_src), dim=1)
            trans_out = self.transition(trans_in)
            prob_plan = []
            n_obs = len(next_obs[0][0])
            for i, no in enumerate(next_obs):
                nno = trans_out[:, i * n_obs:(i + 1) * n_obs]
                plan_in = torch.cat((no, nno), dim=1)
                prob_plan.append(gumbel_softmax(self.planning(plan_in), device=self.device.get_device(), hard=False))
            prob_plan = torch.cat(prob_plan, dim=1)
            prob_src = torch.cat(prob_src, dim=1)
            acs_src = torch.cat(acs_src, dim=1)

            E = acs_src * prob_plan - acs_src * prob_src
            i_rews = E.mean() * torch.ones((1, rewards.shape[1]))
            return i_rews.numpy()


class TrainerJoint(object):
    def __init__(self, variational_joint_empowerment):
        self.transition = variational_joint_empowerment.transition
        self.source = variational_joint_empowerment.source
        self.planning = variational_joint_empowerment.planning
        self.device = variational_joint_empowerment.device

        self.transition_optimizer = Adam(self.transition.parameters(), lr=variational_joint_empowerment.lr)
        self.planning_optimizer = Adam(self.planning.parameters(), lr=variational_joint_empowerment.lr)
        self.source_optimizer = Adam(list(self.source.parameters()) + list(self.planning.parameters()), lr=variational_joint_empowerment.lr)

        self.niter = 0

    def update(self, sample, logger):
        obs, acs, rews, emps, next_obs, dones = sample

        self.transition_optimizer.zero_grad()
        trans_in = torch.cat((*obs, *acs), dim=1)
        next_obs_pred = self.transition(trans_in)
        trans_loss = MSELoss(next_obs_pred, torch.cat(next_obs, dim=1))
        trans_loss.backward()
        self.transition_optimizer.step()

        self.planning_optimizer.zero_grad()
        acs_plan = []
        for o, no in zip(obs, next_obs):
            plan_in = torch.cat((o, no), dim=1)
            acs_plan.append(gumbel_softmax(self.planning(plan_in), device=self.device.get_device(), hard=True))
        acs_plan = torch.cat(acs_plan, dim=1)
        acs_torch = torch.cat(acs, dim=1)
        plan_loss = MSELoss(acs_plan, acs_torch)
        plan_loss.backward()
        self.planning_optimizer.step()

        self.source_optimizer.zero_grad()
        acs_src = []
        prob_src = []
        for no in next_obs:
            acs_src.append(gumbel_softmax(self.source(no), device=self.device.get_device(), hard=True))
            prob_src.append(gumbel_softmax(self.source(no), device=self.device.get_device(), hard=False))
        with torch.no_grad():
            trans_in = torch.cat((*next_obs, *acs_src), dim=1)
            trans_out = self.transition(trans_in)
        prob_plan = []
        n_obs = len(next_obs[0][0])
        for i, no in enumerate(next_obs):
            nno = trans_out[:, i * n_obs:(i + 1) * n_obs]
            plan_in = torch.cat((no, nno), dim=1)
            prob_plan.append(gumbel_softmax(self.planning(plan_in), device=self.device.get_device(), hard=False))
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
                                'plan_loss': plan_loss.detach(),
                                'i_rews': i_rews.detach()},
                               self.niter)
        self.niter += 1


class VariationalJointEmpowerment(VariationalBaseEmpowerment):
    def __init__(self, init_params, lr=0.01):
        super(VariationalJointEmpowerment, self).__init__()
        self.transition = MLPNetwork(init_params['num_in_trans'], init_params['num_out_trans'], recurrent=True)
        self.source = MLPNetwork(init_params['num_in_src'], init_params['num_out_src'], recurrent=True)
        self.planning = MLPNetwork(init_params['num_in_plan'], init_params['num_out_plan'], recurrent=True)
        self.lr = lr

        self.device = Device('cpu')

        self.computer = ComputerJoint(self)
        self.trainer = TrainerJoint(self)

    def compute(self, rewards, next_obs):
        return self.computer.compute(rewards, next_obs)

    def update(self, sample, logger=None):
        return self.trainer.update(sample, logger)

    def prep_training(self, device='gpu'):
        self.transition.train()
        self.source.train()
        self.planning.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)
            self.source = fn(self.source)
            self.planning = fn(self.planning)

        self.device.set_device(device)

    def prep_rollouts(self, device='cpu'):
        self.transition.eval()
        self.source.eval()
        self.planning.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.device.get_device() == device:
            self.transition = fn(self.transition)
            self.source = fn(self.source)
            self.planning = fn(self.planning)

        self.device.set_device(device)

    @classmethod
    def init_from_env(cls, env):
        num_in_source = num_out_source = num_in_planning = \
            num_out_planning = num_in_transition = num_out_transition = 0
        for acsp, obsp in zip(env.action_space, env.observation_space):

            num_in_source = obsp.shape[0]
            num_out_source = acsp.n

            num_in_planning = 2*obsp.shape[0]
            num_out_planning = acsp.n

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
