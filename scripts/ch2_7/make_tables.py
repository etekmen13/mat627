import os

import numpy as np

DATA_DIR = "data/ch2_7"
OUT_DIR = "reports/ch2_7/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def num_sci(x: float) -> str:
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def num_dec(x: float) -> str:
    return f"{float(x):.1f}"


def num_h(x: float) -> str:
    return f"{float(x):.5f}".rstrip("0").rstrip(".")


def num_rate(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{float(x):.6f}"


def load_part1():
    return {
        "x": np.load(os.path.join(DATA_DIR, "part1__x.npy")),
        "approx": np.load(os.path.join(DATA_DIR, "part1__approx.npy")),
        "exact": np.load(os.path.join(DATA_DIR, "part1__exact.npy")),
        "err": np.load(os.path.join(DATA_DIR, "part1__err.npy")),
    }


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def make_part1_table(rows):
    lines = []
    lines.append(r"\section{Part 1}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{7pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"\caption{Finite-difference solution for $-u'' + u = 4e^{-x} - 4xe^{-x}$ with $N=5$.}"
    )
    lines.append(r"\label{tab:part1}")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(r"$x_k$ & $U_k$ & $u(x_k)$ & $u(x_k)-U_k$ \\")
    lines.append(r"\midrule")

    for x, approx, exact, err in zip(
        rows["x"], rows["approx"], rows["exact"], rows["err"]
    ):
        lines.append(
            f"{num_dec(float(x))} & {num_sci(float(approx))} & "
            f"{num_sci(float(exact))} & {num_sci(float(err))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_summary_table(title: str, label: str, problem_tex: str, summary):
    lines = []
    lines.append(rf"\section{{{title}}}")
    lines.append(rf"\noindent {problem_tex}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{7pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{Maximum nodal error for {title.lower()}.}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $h$ & $E_h = \max_k |u(x_k)-U_k|$ & Rate \\")
    lines.append(r"\midrule")

    for n, h, err, rate in zip(
        summary["n"], summary["h"], summary["err"], summary["rate"]
    ):
        lines.append(
            f"{int(round(float(n)))} & {num_h(float(h))} & {num_sci(float(err))} & "
            f"{num_rate(float(rate))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out = []
    out.append("% Auto-generated from data/ch2_7/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("")
    out.append(make_part1_table(load_part1()))
    out.append(
        make_summary_table(
            "Part 2",
            "part2",
            r"Discrete error for $-u'' + u = 4e^{-x} - 4xe^{-x}$ with exact solution $u(x)=x(1-x)e^{-x}$.",
            load_summary("part2"),
        )
    )
    out.append(
        make_summary_table(
            "Part 3",
            "part3",
            r"Discrete error for $-u'' + u = 2 + x - x^2$ with exact solution $u(x)=x(1-x)$.",
            load_summary("part3"),
        )
    )
    out.append(
        make_summary_table(
            "Part 4",
            "part4",
            r"Discrete error for $-u'' + u' + u = -(x-1)(x^2-11)$ using the backward difference $(u(x)-u(x-h))/h$ for $u'$.",
            load_summary("part4"),
        )
    )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
