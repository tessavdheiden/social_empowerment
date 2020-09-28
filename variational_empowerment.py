import torch
from torch.optim import Adam
MSELoss = torch.nn.MSELoss()


from utils.networks import MLPNetwork



class VariationalJointEmpowerment(object):
    def __init__(self, init_params, lr=0.01):
        super(VariationalJointEmpowerment, self).__init__()
        self.transition = MLPNetwork(init_params['num_in_transition'], init_params['num_out_transition'], recurrent=True)
        self.transition_optimizer = Adam(self.transition.parameters(), lr=lr)

    def compute(self, observations):
        pass

    def update(self, sample):
        obs, acs, rews, emps, next_obs, dones = sample

        self.transition_optimizer.zero_grad()
        trans_in = torch.cat((*obs, *acs), dim=1)
        next_obs_pred = self.transition(trans_in)
        trans_loss = MSELoss(next_obs_pred, torch.cat((next_obs), dim=1))
        trans_loss.backward()
        self.transition_optimizer.step()

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

        init_params = {'num_in_source': num_in_source,
                        'num_in_planning': num_in_planning,
                        'num_in_transition': num_in_transition,
                        'num_out_source': num_out_source,
                        'num_out_planning': num_out_planning,
                        'num_out_transition': num_out_transition}

        instance = cls(init_params)
        instance.init_dict = init_params
        return instance



