import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch2_3"
PLOT_DIR = "plots/ch2_3"
OUT_FILE = os.path.join(PLOT_DIR, "plot.png")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    t_exact = np.load(os.path.join(DATA_DIR, "plot__exact_t.npy"))
    y_exact = np.load(os.path.join(DATA_DIR, "plot__exact_y.npy"))
    t_euler = np.load(os.path.join(DATA_DIR, "plot__euler_t.npy"))
    y_euler = np.load(os.path.join(DATA_DIR, "plot__euler_y.npy"))

    plt.figure(figsize=(8, 5))
    plt.plot(t_exact, y_exact, color="black", label="exact solution")
    plt.plot(t_euler, y_euler, "o--", color="red", label="Euler, h=0.25")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Euler approximation versus the exact solution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150)
    plt.close()

    print(f"Saved plot to {OUT_FILE}")


if __name__ == "__main__":
    main()
