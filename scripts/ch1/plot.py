import numpy as np
import sys
import matplotlib.pyplot as plt
import os


dir = sys.argv[1]

out_file = sys.argv[2]


alternating_dir = f"{dir}/Alternating"
reciprocal_dir = f"{dir}/Reciprocal"
fig, axs = plt.subplots(5, 3, figsize=(10, 10))

for i, (a_fname, r_fname) in enumerate(
    zip(os.listdir(alternating_dir), os.listdir(reciprocal_dir))
):
    a_data = np.load(f"{alternating_dir}/{a_fname}")
    r_data = np.load(f"{reciprocal_dir}/{r_fname}")

    x = i % 5
    y = i // 5

    #    axs[x, y].set_yscale("log")
    axs[x, y].plot(a_data, label="Alternating")
    axs[x, y].plot(r_data, label="Reciprocal")
    axs[x, y].set_title(f"X={a_fname.split('.')[0]}", pad=10)

handles, labels = axs[0, 0].get_legend_handles_labels()
axs[3, 2].axis("off")
axs[4, 2].axis("off")
fig.legend(handles, labels, loc="lower right")
fig.tight_layout()
fig.savefig(out_file)
