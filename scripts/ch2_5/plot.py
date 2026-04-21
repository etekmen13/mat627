import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch2_5"
PLOT_DIR = "plots/ch2_5"


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "approx": np.load(os.path.join(DATA_DIR, f"{name}__approx.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def load_plot(name: str):
    return {
        "x": np.load(os.path.join(DATA_DIR, f"plot__{name}__x.npy")),
        "exact": np.load(os.path.join(DATA_DIR, f"plot__{name}__exact.npy")),
        "interp": np.load(os.path.join(DATA_DIR, f"plot__{name}__interp.npy")),
        "nodes_x": np.load(os.path.join(DATA_DIR, f"plot__{name}__nodes_x.npy")),
        "nodes_y": np.load(os.path.join(DATA_DIR, f"plot__{name}__nodes_y.npy")),
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    cases = [
        ("part1", r"$f(x)=x^2e^{-x}$ on $[1,3]$"),
        ("part2", r"$f(x)=x^2e^{-x}$ on $[0,2]$"),
        ("part3", r"$f(x)=\sqrt{x}$ on $[0,1]$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    for ax, (name, title) in zip(axes, cases):
        plot = load_plot(name)
        ax.plot(plot["x"], plot["exact"], color="black", linewidth=2, label="exact integrand")
        ax.plot(
            plot["x"],
            plot["interp"],
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label=r"piecewise linear interpolant",
        )
        ax.fill_between(
            plot["x"],
            0.0,
            plot["interp"],
            color="tab:orange",
            alpha=0.25,
            label="trapezoid-rule area",
        )
        ax.scatter(
            plot["nodes_x"],
            plot["nodes_y"],
            color="tab:blue",
            s=30,
            zorder=3,
            label="mesh points",
        )
        ax.set_title(title + r" with $N=4$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best")

    fig.savefig(os.path.join(PLOT_DIR, "trapezoids.png"), dpi=150)
    plt.close(fig)

    part1 = load_summary("part1")
    part2 = load_summary("part2")
    part3 = load_summary("part3")

    plt.figure(figsize=(8, 5))
    plt.loglog(part1["h"], part1["err"], "o-", linewidth=2, label=r"Part 1: $[1,3]$")
    plt.loglog(part2["h"], part2["err"], "s-", linewidth=2, label=r"Part 2: $[0,2]$")
    plt.loglog(part3["h"], part3["err"], "d-", linewidth=2, label=r"Part 3: $[0,1]$")

    ref2 = part1["err"][-1] * (part1["h"] / part1["h"][-1]) ** 2
    ref4 = part2["err"][-1] * (part2["h"] / part2["h"][-1]) ** 4
    ref32 = part3["err"][-1] * (part3["h"] / part3["h"][-1]) ** 1.5

    plt.loglog(part1["h"], ref2, "--", color="gray", label=r"reference slope $2$")
    plt.loglog(part2["h"], ref4, "-.", color="tab:green", label=r"reference slope $4$")
    plt.loglog(part3["h"], ref32, ":", color="black", label=r"reference slope $3/2$")

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$E_h = |T_N(f)-I(f)|$")
    plt.title("Composite trapezoid-rule error decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "error.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
