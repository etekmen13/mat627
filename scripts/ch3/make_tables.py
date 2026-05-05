import os

import numpy as np

DATA_DIR = "data/ch3"
OUT_DIR = "reports/ch3/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def load_iter(slug: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{slug}__n.npy")),
        "step": np.load(os.path.join(DATA_DIR, f"{slug}__step.npy")),
        "fx_abs": np.load(os.path.join(DATA_DIR, f"{slug}__fx_abs.npy")),
        "x": np.load(os.path.join(DATA_DIR, f"{slug}__x.npy")),
        "error": np.load(os.path.join(DATA_DIR, f"{slug}__error.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{slug}__rate.npy")),
        "status": open(
            os.path.join(DATA_DIR, f"{slug}__status.txt"), encoding="utf-8"
        ).read(),
    }


def load_bracket(slug: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{slug}__n.npy")),
        "a": np.load(os.path.join(DATA_DIR, f"{slug}__a.npy")),
        "b": np.load(os.path.join(DATA_DIR, f"{slug}__b.npy")),
        "x": np.load(os.path.join(DATA_DIR, f"{slug}__x.npy")),
        "half_width": np.load(os.path.join(DATA_DIR, f"{slug}__half_width.npy")),
        "error": np.load(os.path.join(DATA_DIR, f"{slug}__error.npy")),
        "status": open(
            os.path.join(DATA_DIR, f"{slug}__status.txt"), encoding="utf-8"
        ).read(),
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
        "status": open(
            os.path.join(DATA_DIR, "grad_safe__status.txt"), encoding="utf-8"
        ).read(),
    }


def num_sci(x: float) -> str:
    s = f"{float(x):.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def num_rate(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{float(x):.6f}"


def num_step(x: float) -> str:
    return "--" if not np.isfinite(x) else num_sci(float(x))


def sign_label(code: float) -> str:
    if code > 0.5:
        return "positive"
    if code < -0.5:
        return "negative"
    return "uncertain"


def add_status(lines: list[str], status: str):
    lines.append(r"\noindent\textbf{Status:} \texttt{\detokenize{" + status + r"}}")
    lines.append("")


def make_bracket_table(title: str, label: str, caption: str, slug: str, with_error: bool):
    data = load_bracket(slug)
    lines = []
    lines.append(rf"\subsection*{{{title}}}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    if with_error:
        lines.append(r"\begin{tabular}{rrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"$n$ & $a_n$ & $b_n$ & $x_n$ & $\frac{1}{2}(b_n-a_n)$ & $|x_n-\alpha|$ \\")
    else:
        lines.append(r"\begin{tabular}{rrrrr}")
        lines.append(r"\toprule")
        lines.append(r"$n$ & $a_n$ & $b_n$ & $x_n$ & $\frac{1}{2}(b_n-a_n)$ \\")
    lines.append(r"\midrule")

    for row in zip(
        data["n"], data["a"], data["b"], data["x"], data["half_width"], data["error"]
    ):
        n, a, b, x, half_width, error = row
        if with_error:
            lines.append(
                f"{int(round(float(n)))} & {num_sci(a)} & {num_sci(b)} & "
                f"{num_sci(x)} & {num_sci(half_width)} & {num_sci(error)} \\\\"
            )
        else:
            lines.append(
                f"{int(round(float(n)))} & {num_sci(a)} & {num_sci(b)} & "
                f"{num_sci(x)} & {num_sci(half_width)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    add_status(lines, data["status"])
    return "\n".join(lines)


def make_iter_table(title: str, label: str, caption: str, slug: str, with_error: bool):
    data = load_iter(slug)
    lines = []
    lines.append(rf"\subsection*{{{title}}}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    if with_error:
        lines.append(r"\begin{tabular}{rrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"$n$ & $|x_n-x_{n-1}|$ & $|f(x_n)|$ & $x_n$ & $|x_n-\alpha|$ & Rate \\")
    else:
        lines.append(r"\begin{tabular}{rrrrr}")
        lines.append(r"\toprule")
        lines.append(r"$n$ & $|x_n-x_{n-1}|$ & $|f(x_n)|$ & $x_n$ & Rate \\")
    lines.append(r"\midrule")

    for row in zip(
        data["n"], data["step"], data["fx_abs"], data["x"], data["error"], data["rate"]
    ):
        n, step, fx_abs, x, error, rate = row
        if with_error:
            lines.append(
                f"{int(round(float(n)))} & {num_step(step)} & {num_sci(fx_abs)} & "
                f"{num_sci(x)} & {num_sci(error)} & {num_rate(rate)} \\\\"
            )
        else:
            lines.append(
                f"{int(round(float(n)))} & {num_step(step)} & {num_sci(fx_abs)} & "
                f"{num_sci(x)} & {num_rate(rate)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    add_status(lines, data["status"])
    return "\n".join(lines)


def make_grad_scan_table():
    data = load_grad_scan()
    lines = []
    lines.append(r"\subsection*{Naive Bisection Scan}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(
        r"\caption{Naive bisection on the degree-9 polynomial for slightly perturbed intervals near $[1.92,2.08]$.}"
    )
    lines.append(r"\label{tab:grad-scan}")
    lines.append(r"\begin{tabular}{rrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$a$ & $b$ & $x_N$ & $N$ & $|p(x_N)|$ \\")
    lines.append(r"\midrule")

    for row in zip(
        data["a"], data["b"], data["root"], data["iterations"], data["residual"]
    ):
        a, b, root, iterations, residual = row
        lines.append(
            f"{num_sci(a)} & {num_sci(b)} & {num_sci(root)} & "
            f"{int(round(float(iterations)))} & {num_sci(residual)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_grad_safe_table():
    data = load_grad_safe()
    lines = []
    lines.append(r"\subsection*{Safeguarded Bisection}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(
        r"\caption{Safeguarded bisection with Horner error bounds on $[1.0,2.5]$.}"
    )
    lines.append(r"\label{tab:grad-safe}")
    lines.append(r"\begin{tabular}{rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"$n$ & $a_n$ & $b_n$ & $x_n$ & $\frac{1}{2}(b_n-a_n)$ & $|\widetilde{p}(x_n)|$ & $d_n$ & sign \\"
    )
    lines.append(r"\midrule")

    for row in zip(
        data["n"],
        data["a"],
        data["b"],
        data["x"],
        data["half_width"],
        np.abs(data["value"]),
        data["bound"],
        data["sign_code"],
    ):
        n, a, b, x, half_width, value_abs, bound, sign_code = row
        lines.append(
            f"{int(round(float(n)))} & {num_sci(a)} & {num_sci(b)} & {num_sci(x)} & "
            f"{num_sci(half_width)} & {num_sci(value_abs)} & {num_sci(bound)} & "
            f"{sign_label(float(sign_code))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    add_status(lines, data["status"])
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out = []
    out.append("% Auto-generated from data/ch3/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("")

    out.append(r"\section{Experiment 1}")
    out.append(r"\subsection*{Unknown Negative Root}")
    out.append(
        make_bracket_table(
            "Bisection",
            "exp1-neg-bisection",
            r"Bisection on $[-3,-2]$ for the polynomial in Experiment 1.",
            "exp1_neg_bisection",
            False,
        )
    )
    out.append(
        make_iter_table(
            "Newton",
            "exp1-neg-newton",
            r"Newton's method from $x_0=-2$ for the negative root.",
            "exp1_neg_newton",
            False,
        )
    )
    out.append(
        make_iter_table(
            "Secant",
            "exp1-neg-secant",
            r"Secant method from $x_0=-3$, $x_1=-2$ for the negative root.",
            "exp1_neg_secant",
            False,
        )
    )
    out.append(
        make_bracket_table(
            "Hybrid",
            "exp1-neg-hybrid",
            r"Hybrid method on $[-3,-2]$ for the negative root.",
            "exp1_neg_hybrid",
            False,
        )
    )
    out.append(
        make_iter_table(
            "Super Halley",
            "exp1-neg-halley",
            r"Super Halley method from $x_0=-2$ for the negative root.",
            "exp1_neg_super_halley",
            False,
        )
    )

    out.append(r"\subsection*{Double Root at $\alpha=2.5$}")
    out.append(
        make_iter_table(
            "Newton",
            "exp1-double-newton",
            r"Newton's method from $x_0=2$ for the double root.",
            "exp1_double_newton",
            True,
        )
    )
    out.append(
        make_iter_table(
            "Secant",
            "exp1-double-secant",
            r"Secant method from $x_0=2$, $x_1=2.1$ for the double root.",
            "exp1_double_secant",
            True,
        )
    )
    out.append(
        make_iter_table(
            "Super Halley",
            "exp1-double-halley",
            r"Super Halley method from $x_0=2$ for the double root.",
            "exp1_double_super_halley",
            True,
        )
    )

    out.append(r"\section{Experiment 2}")
    out.append(
        make_iter_table(
            "Chord",
            "exp2-chord",
            r"Chord method for $f(x)=2-e^x$ with $x_0=0$ and $M=20$.",
            "exp2_chord",
            True,
        )
    )
    out.append(
        make_iter_table(
            "Secant",
            "exp2-secant",
            r"Secant method for $f(x)=2-e^x$ with $x_0=0$, $x_1=2$, and $M=20$.",
            "exp2_secant",
            True,
        )
    )
    out.append(
        make_iter_table(
            "Newton",
            "exp2-newton",
            r"Newton's method for $f(x)=2-e^x$ with $x_0=0$ and $M=20$.",
            "exp2_newton",
            True,
        )
    )
    out.append(
        make_iter_table(
            "Super Halley",
            "exp2-halley",
            r"Super Halley method for $f(x)=2-e^x$ with $x_0=0$ and $M=20$.",
            "exp2_super_halley",
            True,
        )
    )

    out.append(r"\section{Experiment 3}")
    out.append(
        make_iter_table(
            r"Newton from $x_0=1$",
            "exp3-newton-pos",
            r"Newton's method for $f(x)=10xe^{-x^2}$ with $x_0=1$, $TOL=10^{-4}$, and $M=30$.",
            "exp3_newton_pos",
            True,
        )
    )
    out.append(
        make_iter_table(
            r"Newton from $x_0=-0.705$",
            "exp3-newton-near",
            r"Newton's method for $f(x)=10xe^{-x^2}$ with $x_0=-0.705$, $TOL=10^{-4}$, and $M=30$.",
            "exp3_newton_near_singular",
            True,
        )
    )
    out.append(
        make_bracket_table(
            r"Hybrid on $[-0.705,1]$",
            "exp3-hybrid-1",
            r"Hybrid method for $f(x)=10xe^{-x^2}$ on $[-0.705,1]$.",
            "exp3_hybrid_bracket1",
            True,
        )
    )
    out.append(
        make_bracket_table(
            r"Hybrid on $[-0.71,2.71]$",
            "exp3-hybrid-2",
            r"Hybrid method for $f(x)=10xe^{-x^2}$ on $[-0.71,2.71]$.",
            "exp3_hybrid_bracket2",
            True,
        )
    )

    out.append(r"\section{Graduate Section}")
    out.append(make_grad_scan_table())
    out.append(make_grad_safe_table())

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
