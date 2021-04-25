from empowerment.base_empowerment import BaseEmpowerment


class DummyEmpowerment(BaseEmpowerment):
    def __init__(self, agents):
        super(DummyEmpowerment, self).__init__()
        self.agents = agents

    def compute(self, reward, next_obs):
        return reward