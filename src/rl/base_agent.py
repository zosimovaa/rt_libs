import numpy as np
class Agent(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def train(self):
        raise NotImplementedError

    def save_model(self, path="./"):
        pass

    def load_model(self, path):
        pass

    def _sample_transformer(self, state):
        return list(map(lambda p: np.expand_dims(p, 0), state))

    def _batch_transformer(self, batch):
        return list(map(np.array, zip(*batch)))