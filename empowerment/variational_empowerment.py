class VariationalBaseEmpowerment(object):
    def __init__(self):
        pass

    def compute(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def prep_training(self):
        raise NotImplementedError

    def prep_rollouts(self):
        raise NotImplementedError

