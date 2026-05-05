import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch4"
PLOT_DIR = "plots/ch4"


def load_summary(prefix: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{prefix}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{prefix}__h.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{prefix}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{prefix}__rate.npy")),
    }


def load_series(prefix: str, *names: str):
    return {
        name: np.load(os.path.join(DATA_DIR, f"{prefix}__{name}.npy"))
        for name in names
    }


def positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), np.finfo(float).tiny)


def plot_experiment_1():
    summaries = [
        ("exp1__lagrange_uniform", "Lagrange (uniform)", "o-"),
        ("exp1__lagrange_chebyshev", "Lagrange (Chebyshev)", "s-"),
        ("exp1__hermite", "Hermite", "^-"),
        ("exp1__piecewise_quadratic", "Piecewise quadratic", "d-"),
        ("exp1__bspline", "Cubic B-spline", "x-"),
    ]

    plt.figure(figsize=(8.5, 5.4))
    for prefix, label, style in summaries:
        data = load_summary(prefix)
        plt.loglog(data["h"], positive(data["err"]), style, linewidth=2, label=label)

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$\max_{z \in D}|p(z)-f(z)|$")
    plt.title(r"Experiment 1: $\sin(x)$ interpolation error")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp1_convergence.png"), dpi=150)
    plt.close()


def plot_experiment_2():
    approx = load_series(
        "exp2_plot",
        "x",
        "exact",
        "lagrange",
        "piecewise_cubic",
        "bspline",
    )
    err = load_series(
        "exp2_error",
        "x",
        "lagrange",
        "piecewise_cubic",
        "bspline",
    )

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(approx["x"], approx["exact"], color="black", linewidth=2, label="exact")
    plt.plot(approx["x"], approx["lagrange"], linewidth=2, label="Lagrange")
    plt.plot(
        approx["x"],
        approx["piecewise_cubic"],
        linewidth=2,
        label="Piecewise cubic",
    )
    plt.plot(approx["x"], approx["bspline"], linewidth=2, label="Cubic B-spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Experiment 2: approximations for $N=6$")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp2_approximation.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(err["x"], err["lagrange"], linewidth=2, label="Lagrange")
    plt.plot(err["x"], err["piecewise_cubic"], linewidth=2, label="Piecewise cubic")
    plt.plot(err["x"], err["bspline"], linewidth=2, label="Cubic B-spline")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel(r"$p(x)-f(x)$")
    plt.title(r"Experiment 2: error functions for $N=12$")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp2_error.png"), dpi=150)
    plt.close()


def plot_experiment_3(tag: str, title_suffix: str):
    approx = load_series(
        f"exp3_{tag}_plot",
        "x",
        "exact",
        "lagrange_uniform",
        "lagrange_chebyshev",
        "piecewise_cubic",
        "bspline",
    )
    err = load_series(
        f"exp3_{tag}_error",
        "x",
        "lagrange_uniform",
        "lagrange_chebyshev",
        "piecewise_cubic",
        "bspline",
    )

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(approx["x"], approx["exact"], color="black", linewidth=2, label="exact")
    plt.plot(
        approx["x"],
        approx["lagrange_uniform"],
        linewidth=2,
        label="Lagrange (uniform)",
    )
    plt.plot(
        approx["x"],
        approx["lagrange_chebyshev"],
        linewidth=2,
        label="Lagrange (Chebyshev)",
    )
    plt.plot(
        approx["x"],
        approx["piecewise_cubic"],
        linewidth=2,
        label="Piecewise cubic",
    )
    plt.plot(approx["x"], approx["bspline"], linewidth=2, label="Cubic B-spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Experiment 3: approximations for N=6, {title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"exp3_{tag}_approximation.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(
        err["x"],
        err["lagrange_uniform"],
        linewidth=2,
        label="Lagrange (uniform)",
    )
    plt.plot(
        err["x"],
        err["lagrange_chebyshev"],
        linewidth=2,
        label="Lagrange (Chebyshev)",
    )
    plt.plot(err["x"], err["piecewise_cubic"], linewidth=2, label="Piecewise cubic")
    plt.plot(err["x"], err["bspline"], linewidth=2, label="Cubic B-spline")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel(r"$p(x)-f(x)$")
    plt.title(f"Experiment 3: error functions for N=12, {title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"exp3_{tag}_error.png"), dpi=150)
    plt.close()


def plot_graduate():
    approx = load_series("grad_plot", "x", "exact", "approx")
    err = load_series("grad_error", "x", "err")

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(approx["x"], approx["exact"], color="black", linewidth=2, label="exact")
    plt.plot(
        approx["x"],
        approx["approx"],
        color="tab:red",
        linewidth=2,
        label=r"collocation spline ($N=5$)",
    )
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Graduate section: collocation approximation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "grad_approximation.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(err["x"], err["err"], color="tab:blue", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel(r"$u_h(x)-u(x)$")
    plt.title(r"Graduate section: error function for $N=20$")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "grad_error.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3("b1", r"$b=1$")
    plot_experiment_3("b4", r"$b=4$")
    plot_graduate()
    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
