import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "data/ch2_1"
outfile = "plots/ch2_1/plot.png"
colors = ["red", "blue", "black"]
x = np.linspace(1.92, 2.08, num=100_000)
for i, fname in enumerate(reversed(os.listdir(data_dir))):
    data = np.load(f"{data_dir}/{fname}")
    plt.plot(x, data, label=f"{fname}", color=colors[i])


plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(outfile)
