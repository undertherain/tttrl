import numpy as np


class AgentRandom:
    def move(self, field):
        zeros = np.argwhere(field[0] == 0)
        zeros = zeros.reshape(-1)
        mv = np.random.choice(zeros)
        return mv
