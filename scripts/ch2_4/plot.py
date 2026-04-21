import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch2_4"
PLOT_DIR = "plots/ch2_4"


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    x = np.load(os.path.join(DATA_DIR, "plot__x.npy"))
    exact = np.load(os.path.join(DATA_DIR, "plot__exact.npy"))
    approx = np.load(os.path.join(DATA_DIR, "plot__approx.npy"))
    nodes_x = np.load(os.path.join(DATA_DIR, "plot__nodes_x.npy"))
    nodes_y = np.load(os.path.join(DATA_DIR, "plot__nodes_y.npy"))

    plt.figure(figsize=(8, 5))
    plt.plot(x, exact, color="black", linewidth=2, label=r"$f(x)=x^{1/3}$")
    plt.plot(x, approx, color="tab:red", linestyle="--", linewidth=2, label=r"$q_4(x)$")
    plt.scatter(nodes_x, nodes_y, color="tab:blue", s=35, zorder=3, label="mesh points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Piecewise linear interpolation of $x^{1/3}$ on $[0,2]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "approximation.png"), dpi=150)
    plt.close()

    smooth = load_summary("smooth")
    singular = load_summary("singular")

    plt.figure(figsize=(8, 5))
    plt.loglog(smooth["h"], smooth["err"], "o-", linewidth=2, label=r"$[1,2]$")
    plt.loglog(singular["h"], singular["err"], "s-", linewidth=2, label=r"$[0,1]$")

    smooth_ref = smooth["err"][-1] * (smooth["h"] / smooth["h"][-1]) ** 2
    singular_ref = singular["err"][-1] * (singular["h"] / singular["h"][-1]) ** (1 / 3)

    plt.loglog(smooth["h"], smooth_ref, "--", color="gray", label=r"reference slope $2$")
    plt.loglog(
        singular["h"],
        singular_ref,
        ":",
        color="black",
        label=r"reference slope $1/3$",
    )

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$E_h$")
    plt.title("Maximum interpolation error on the fine mesh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "error.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
