import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from field import Field
from tournament import eval_players
from agent_random import AgentRandom
import os
import json


path_results = "results"


def save_data_json(data, name_file):
    s = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
    f = open(name_file, 'w')
    print(s, file=f)
    f.close()


class Policy(chainer.Chain):
    def __init__(self, train=True):
        super().__init__(
            l1=L.Linear(None, 9, initialW=I.Uniform(1. / 9)),
            l2=L.Linear(None, 9, initialW=I.Uniform(1. / 9))
        )
        self.train = train

    def __call__(self, x):
        h = x
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h


class AgentNeural:
    def __init__(self, policy):
        self.policy = policy
        self.history = []

    def reset(self):
        self.history = []

    def sample_move_from_proba(self, proba):
        moves = []
        for i in range(proba.shape[0]):
            p = proba.data[i]
            p /= p.sum()
            moves.append(np.random.choice(a=9, p=p))
        moves = np.array(moves, dtype=np.int32)
        return moves

    def move(self, field):
        move_proba = self.policy(field)
        mask = (field > 0) | (field < 0)
        mask = mask.astype(np.int32)
        mask = 1 - mask
        proba = F.softmax(move_proba)
        proba *= mask

        moves = self.sample_move_from_proba(proba)
        self.history.append((move_proba, moves))
        return moves[0]

    def loss_victory(self):
        loss_total = chainer.Variable(np.array(0, dtype=np.float32))
        for proba, move in self.history:
            loss = F.softmax_cross_entropy(proba, move)
            loss_total += loss
        return loss_total

    def loss_defeat(self):
        loss_total = chainer.Variable(np.array(0, dtype=np.float32))
        for proba, move in self.history:
            loss = -F.softmax_cross_entropy(proba, move)
            loss_total += loss
        return loss_total


def train(players, optimizer):
    players[0].reset()
    players[1].reset()
    field = Field()
    for turn in range(9):
        id_player = turn % 2
        move = players[id_player].move(field.field)
        field.field[0, move] = 1
        status = field.eval()

        if id_player == 1:
            status = - status

        if status[0] != 0:
            loss = players[id_player].loss_victory() + players[1 - id_player].loss_defeat()
            return loss
        field.field = -field.field
    return None


def main():
    policy = Policy()
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(policy)
    players = [AgentNeural(policy), AgentNeural(policy)]
    progress = {}
    progress["win0"] = []
    progress["loss"] = []

    for epoch in range(100):
        loss = train(players, optimizer)
        if loss is not None:
            policy.cleargrads()
            loss.backward()
            optimizer.update()
        stats = eval_players([players[0], AgentRandom()])
        print(epoch, stats)
        progress["win0"].append(stats[0]["win"])
        if epoch % 3 == 0:
            save_data_json(progress, os.path.join(path_results, "progress.json"))
#            plt.figure()
#            plt.plot(np.arange(len(progress)), progress)
#            plt.xlabel("iterations")
#            plt.ylabel("wins")
#            plt.savefig("progress.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
