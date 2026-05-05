import os

import numpy as np

DATA_DIR = "data/ch4"
OUT_DIR = "reports/ch4/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def load_summary(prefix: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{prefix}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{prefix}__h.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{prefix}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{prefix}__rate.npy")),
    }


def num_sci(x: float) -> str:
    s = f"{float(x):.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def num_h(x: float) -> str:
    return f"{float(x):.5f}".rstrip("0").rstrip(".")


def num_rate(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{float(x):.6f}"


def make_error_table(title: str, label: str, prefixes: list[tuple[str, str]]):
    rows = [load_summary(prefix) for prefix, _ in prefixes]
    n_values = rows[0]["n"]

    lines = []
    lines.append(rf"\section{{{title}}}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(rf"\caption{{Maximum fine-grid interpolation errors for {title.lower()}.}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append("".join([r"\begin{tabular}{r", "r" * len(prefixes), "}"]))
    lines.append(r"\toprule")
    header = " & ".join([r"$N$"] + [name for _, name in prefixes]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for idx, n in enumerate(n_values):
        values = [num_sci(float(row["err"][idx])) for row in rows]
        lines.append(
            f"{int(round(float(n)))} & " + " & ".join(values) + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_grad_table():
    data = load_summary("grad")
    lines = []
    lines.append(r"\section{Graduate Section}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(
        r"\caption{Fine-grid collocation errors for $-u'' + u = 4e^{-x} - 4xe^{-x}$ with exact solution $u(x)=x(1-x)e^{-x}$.}"
    )
    lines.append(r"\label{tab:grad}")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $h$ & $E_h = \max_{z \in D}|u_h(z)-u(z)|$ & Rate \\")
    lines.append(r"\midrule")

    for n, h, err, rate in zip(
        data["n"], data["h"], data["err"], data["rate"], strict=True
    ):
        lines.append(
            f"{int(round(float(n)))} & {num_h(float(h))} & {num_sci(float(err))} & {num_rate(float(rate))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out = []
    out.append("% Auto-generated from data/ch4/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("")
    out.append(
        make_error_table(
            "Experiment 1",
            "exp1",
            [
                ("exp1__lagrange_uniform", "Lagrange 1"),
                ("exp1__lagrange_chebyshev", "Lagrange 2"),
                ("exp1__hermite", "Hermite"),
                ("exp1__piecewise_quadratic", "Piecewise Quadratic"),
                ("exp1__bspline", "Cubic Splines"),
            ],
        )
    )
    out.append(
        make_error_table(
            "Experiment 2",
            "exp2",
            [
                ("exp2__lagrange", "Lagrange"),
                ("exp2__piecewise_cubic", "Piecewise Cubic"),
                ("exp2__bspline", "Cubic Splines"),
            ],
        )
    )
    out.append(
        make_error_table(
            "Experiment 3 (b=1)",
            "exp3-b1",
            [
                ("exp3_b1__lagrange_uniform", "Lagrange 1"),
                ("exp3_b1__lagrange_chebyshev", "Lagrange 2"),
                ("exp3_b1__piecewise_cubic", "Piecewise Cubic"),
                ("exp3_b1__bspline", "Cubic Splines"),
            ],
        )
    )
    out.append(
        make_error_table(
            "Experiment 3 (b=4)",
            "exp3-b4",
            [
                ("exp3_b4__lagrange_uniform", "Lagrange 1"),
                ("exp3_b4__lagrange_chebyshev", "Lagrange 2"),
                ("exp3_b4__piecewise_cubic", "Piecewise Cubic"),
                ("exp3_b4__bspline", "Cubic Splines"),
            ],
        )
    )
    out.append(make_grad_table())

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
