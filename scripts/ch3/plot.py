import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/ch3"
PLOT_DIR = "plots/ch3"


def load_iter(slug: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{slug}__n.npy")),
        "step": np.load(os.path.join(DATA_DIR, f"{slug}__step.npy")),
        "fx_abs": np.load(os.path.join(DATA_DIR, f"{slug}__fx_abs.npy")),
        "x": np.load(os.path.join(DATA_DIR, f"{slug}__x.npy")),
        "error": np.load(os.path.join(DATA_DIR, f"{slug}__error.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{slug}__rate.npy")),
    }


def load_bracket(slug: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{slug}__n.npy")),
        "a": np.load(os.path.join(DATA_DIR, f"{slug}__a.npy")),
        "b": np.load(os.path.join(DATA_DIR, f"{slug}__b.npy")),
        "x": np.load(os.path.join(DATA_DIR, f"{slug}__x.npy")),
        "half_width": np.load(os.path.join(DATA_DIR, f"{slug}__half_width.npy")),
        "error": np.load(os.path.join(DATA_DIR, f"{slug}__error.npy")),
    }


def load_xy(slug: str):
    return {
        "x": np.load(os.path.join(DATA_DIR, f"{slug}__x.npy")),
        "y": np.load(os.path.join(DATA_DIR, f"{slug}__y.npy")),
    }


def load_grad_scan():
    return {
        "a": np.load(os.path.join(DATA_DIR, "grad_scan__a.npy")),
        "b": np.load(os.path.join(DATA_DIR, "grad_scan__b.npy")),
        "root": np.load(os.path.join(DATA_DIR, "grad_scan__root.npy")),
        "iterations": np.load(os.path.join(DATA_DIR, "grad_scan__iterations.npy")),
        "residual": np.load(os.path.join(DATA_DIR, "grad_scan__residual.npy")),
    }


def load_grad_safe():
    return {
        "n": np.load(os.path.join(DATA_DIR, "grad_safe__n.npy")),
        "a": np.load(os.path.join(DATA_DIR, "grad_safe__a.npy")),
        "b": np.load(os.path.join(DATA_DIR, "grad_safe__b.npy")),
        "x": np.load(os.path.join(DATA_DIR, "grad_safe__x.npy")),
        "half_width": np.load(os.path.join(DATA_DIR, "grad_safe__half_width.npy")),
        "error": np.load(os.path.join(DATA_DIR, "grad_safe__error.npy")),
        "value": np.load(os.path.join(DATA_DIR, "grad_safe__value.npy")),
        "bound": np.load(os.path.join(DATA_DIR, "grad_safe__bound.npy")),
        "sign_code": np.load(os.path.join(DATA_DIR, "grad_safe__sign_code.npy")),
    }


def positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), np.finfo(float).tiny)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    exp1_function = load_xy("exp1_function")
    neg_root = load_bracket("exp1_neg_bisection")["x"][-1]
    double_root = load_iter("exp1_double_newton")["x"][-1]

    plt.figure(figsize=(8, 5))
    plt.plot(exp1_function["x"], exp1_function["y"], color="tab:blue", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(neg_root, color="tab:red", linestyle="--", linewidth=1.5, label="negative root")
    plt.axvline(
        double_root,
        color="tab:green",
        linestyle=":",
        linewidth=1.8,
        label="double root",
    )
    plt.xlim(-3.0, 9.0)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Experiment 1 polynomial")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp1_function.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for slug, label, marker in [
        ("exp1_double_newton", "Newton", "o-"),
        ("exp1_double_secant", "Secant", "s-"),
        ("exp1_double_super_halley", "Super Halley", "d-"),
    ]:
        run = load_iter(slug)
        plt.semilogy(run["n"], positive(run["error"]), marker, linewidth=2, label=label)

    plt.xlabel("Iteration n")
    plt.ylabel(r"$|x_n-\alpha|$")
    plt.title("Experiment 1: convergence toward the double root")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp1_multiple_root.png"), dpi=150)
    plt.close()

    exp2_function = load_xy("exp2_function")
    plt.figure(figsize=(8, 5))
    plt.plot(exp2_function["x"], exp2_function["y"], color="tab:purple", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(np.log(2.0), color="tab:red", linestyle="--", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(r"Experiment 2 function $f(x)=2-e^x$")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp2_function.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for slug, label, marker in [
        ("exp2_chord", "Chord", "o-"),
        ("exp2_secant", "Secant", "s-"),
        ("exp2_newton", "Newton", "^-"),
        ("exp2_super_halley", "Super Halley", "d-"),
    ]:
        run = load_iter(slug)
        plt.semilogy(run["n"], positive(run["error"]), marker, linewidth=2, label=label)

    plt.xlabel("Iteration n")
    plt.ylabel(r"$|x_n-\alpha|$")
    plt.title("Experiment 2: error decay by method")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp2_rates.png"), dpi=150)
    plt.close()

    exp3_function = load_xy("exp3_function")
    plt.figure(figsize=(8, 5))
    plt.plot(exp3_function["x"], exp3_function["y"], color="tab:orange", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(r"Experiment 3 function $f(x)=10xe^{-x^2}$")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp3_function.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for slug, label, marker in [
        ("exp3_newton_pos", r"Newton, $x_0=1$", "o-"),
        ("exp3_newton_near_singular", r"Newton, $x_0=-0.705$", "s-"),
        ("exp3_hybrid_bracket1", r"Hybrid, $[-0.705,1]$", "^-"),
        ("exp3_hybrid_bracket2", r"Hybrid, $[-0.71,2.71]$", "d-"),
    ]:
        run = load_iter(slug) if "newton" in slug else load_bracket(slug)
        plt.semilogy(run["n"], positive(run["error"]), marker, linewidth=2, label=label)

    plt.xlabel("Iteration n")
    plt.ylabel(r"$|x_n|$")
    plt.title("Experiment 3: local and safeguarded convergence")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "exp3_locality.png"), dpi=150)
    plt.close()

    grad_scan = load_grad_scan()
    grad_safe = load_grad_safe()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(grad_scan["a"], grad_scan["root"], "o-", linewidth=2, color="tab:blue")
    axes[0].axhline(2.0, color="tab:red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Left endpoint a")
    axes[0].set_ylabel("Computed bisection root")
    axes[0].set_title("Naive bisection sensitivity")
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(
        grad_safe["n"],
        positive(grad_safe["value"]),
        "o-",
        linewidth=2,
        label=r"$|\widetilde{p}(x_n)|$",
    )
    axes[1].semilogy(
        grad_safe["n"],
        positive(grad_safe["bound"]),
        "s--",
        linewidth=2,
        label="Horner error bound",
    )
    axes[1].set_xlabel("Iteration n")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Safeguarded stopping test")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "grad_instability.png"), dpi=150)
    plt.close(fig)

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
