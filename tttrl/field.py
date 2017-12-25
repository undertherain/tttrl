import numpy as np


class Field:
    def __init__(self):
        self.field = np.zeros((3, 3), dtype=np.float32)
        # field = np.array([field,field])
        self.field = self.field[np.newaxis, :]
        self.field = self.field.reshape(self.field.shape[0], -1)

    def eval_single(field):
        if 3 in field.sum(axis=0):
            return 1
        if 3 in field.sum(axis=1):
            return 1
        if 3 == field.trace():
            return 1
        if 3 == np.flipud(field).trace():
            return 1
        if -3 in field.sum(axis=0):
            return -1
        if -3 in field.sum(axis=1):
            return -1
        if -3 == field.trace():
            return -1
        if -3 == np.flipud(field).trace():
            return -1
        return 0

    def eval(self):
        result = []
        for i in range(self.field.shape[0]):
            result.append(Field.eval_single(self.field[i].reshape(3, 3)))
        return np.array(result)

    def to_str(self, id_player):
        field = self.field[0].reshape(3, 3)
        if id_player == 1:
            field = - field
        result = ""
        for i in range(3):
            for j in range(3):
                if field[i, j] > 0:
                    result += 'x'
                else:
                    if field[i, j] < 0:
                        result += 'o'
                    else:
                        result += '.'
            result += "\n"
        return result
