class BaseEmpowerment(object):
    def __init__(self):
        pass

    def compute(self, reward, next_obs):
        return NotImplementedError

    def update(self, sample, logger):
        pass

    def prep_training(self, device):
        pass

    def prep_rollouts(self, device):
        pass