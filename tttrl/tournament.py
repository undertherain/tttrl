import numpy as np
from agent_random import AgentRandom
from field import Field


def play_game(players, verbose=False):
    field = Field()
    for turn in range(9):
        id_player = turn % 2
        move = players[id_player].move(field.field)
        field.field[0, move] = 1
        status = field.eval()

        if id_player == 1:
            status = - status

        s = field.to_str(id_player)
        if verbose:
            print(s)

        if status[0] > 0:
            if verbose:
                print("Victory of x!")
            return 1
        if status[0] < 0:
            if verbose:
                print("victory of o!")
            return -1
        field.field = -field.field
    return 0


def eval_players(players):
    cnt_games = 220
    stats = {0: {"win": 0}, 1: {"win": 0}}
    for i in range(cnt_games):
        result = play_game(players)
        if result == 1:
            stats[0]["win"] += 1
        if result == -1:
            stats[1]["win"] += 1
    stats[0]["win"] /= cnt_games
    stats[1]["win"] /= cnt_games
    return stats


def main():
    play_game([AgentRandom(), AgentRandom()], verbose=True)
    stats = eval_players([AgentRandom(), AgentRandom()])
    print(stats)


if __name__ == "__main__":
    main()
