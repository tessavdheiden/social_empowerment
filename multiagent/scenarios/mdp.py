class BaseMDP(object):
    def __init__(self, n_agents, dims, n_step):
        self.T = None
        self.Tn = None
        self.configurations = None

