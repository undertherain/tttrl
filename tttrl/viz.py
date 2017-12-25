import numpy as np
import json
import matplotlib as mpl
from matplotlib import pyplot as plt


with open("./results/progress.json") as f:
    data=json.load(f)["win0"]

plt.plot(np.arange(len(data)),data)
plt.xlabel("epochs")
plt.ylabel("winning rate")
plt.show()
plt.savefig("win_rate.pdf")
