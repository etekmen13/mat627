import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch2_7"
PLOT_DIR = "plots/ch2_7"


def load_part1():
    return {
        "x": np.load(os.path.join(DATA_DIR, "plot__x.npy")),
        "exact": np.load(os.path.join(DATA_DIR, "plot__exact.npy")),
        "nodes_x": np.load(os.path.join(DATA_DIR, "plot__nodes_x.npy")),
        "nodes_y": np.load(os.path.join(DATA_DIR, "plot__nodes_y.npy")),
    }


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    part1 = load_part1()
    plt.figure(figsize=(8, 5))
    plt.plot(part1["x"], part1["exact"], color="black", linewidth=2, label="exact solution")
    plt.plot(
        part1["nodes_x"],
        part1["nodes_y"],
        "o--",
        color="tab:red",
        linewidth=2,
        markersize=6,
        label=r"finite-difference approximation ($N=5$)",
    )
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Part 1 approximation versus exact solution")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "approximation.png"), dpi=150)
    plt.close()

    part2 = load_summary("part2")
    part3 = load_summary("part3")
    part4 = load_summary("part4")

    plt.figure(figsize=(8, 5))
    tiny = np.finfo(float).tiny
    plt.loglog(
        part2["h"],
        np.maximum(part2["err"], tiny),
        "o-",
        linewidth=2,
        label=r"Part 2: $-u'' + u = f$",
    )
    plt.loglog(
        part3["h"],
        np.maximum(part3["err"], tiny),
        "s-",
        linewidth=2,
        label=r"Part 3: quadratic exact solution",
    )
    plt.loglog(
        part4["h"],
        np.maximum(part4["err"], tiny),
        "d-",
        linewidth=2,
        label=r"Part 4: backward difference for $u'$",
    )

    ref2 = part2["err"][-1] * (part2["h"] / part2["h"][-1]) ** 2
    ref1 = part4["err"][-1] * (part4["h"] / part4["h"][-1]) ** 1

    plt.loglog(part2["h"], ref2, "--", color="gray", label=r"reference slope $2$")
    plt.loglog(part4["h"], ref1, ":", color="black", label=r"reference slope $1$")

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$E_h = \max_k |u(x_k) - U_k|$")
    plt.title("Finite-difference error decay")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "error.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
