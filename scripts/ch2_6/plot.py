import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch2_6"
PLOT_DIR = "plots/ch2_6"
N_VALUES = [3, 10, 25, 100]


def load_summary():
    return {
        "n": np.load(os.path.join(DATA_DIR, "summary__n.npy")),
        "en": np.load(os.path.join(DATA_DIR, "summary__en.npy")),
    }


def load_residual_profile(n: int):
    return {
        "i": np.load(os.path.join(DATA_DIR, f"residual__n{n}__i.npy")),
        "abs": np.load(os.path.join(DATA_DIR, f"residual__n{n}__abs.npy")),
    }


def load_solution(n: int):
    return {
        "i": np.load(os.path.join(DATA_DIR, f"solution__n{n}__i.npy")),
        "x": np.load(os.path.join(DATA_DIR, f"solution__n{n}__x.npy")),
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for n in N_VALUES:
        profile = load_residual_profile(n)
        plt.semilogy(
            profile["i"],
            np.maximum(profile["abs"], np.finfo(float).tiny),
            marker="o",
            linewidth=1.8,
            label=rf"$n={n}$",
        )

    plt.xlabel("i")
    plt.ylabel(r"$|r_i|$")
    plt.title("Residual magnitude by component")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residual_profiles.png"), dpi=150)
    plt.close()

    summary = load_summary()
    plt.figure(figsize=(8, 5))
    plt.loglog(summary["n"], summary["en"], "o-", linewidth=2, color="tab:red")
    plt.xlabel("n")
    plt.ylabel(r"$e_n = \|r\|_\infty$")
    plt.title("Maximum residual component versus system size")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "max_residual.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for n in N_VALUES:
        solution = load_solution(n)
        plt.plot(solution["i"], solution["x"], linewidth=1.8, label=rf"$n={n}$")

    plt.xlabel("i")
    plt.ylabel(r"$x_i$")
    plt.title("Computed solution components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "solution_profiles.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
