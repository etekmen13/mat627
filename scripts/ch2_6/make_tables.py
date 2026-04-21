import os

import numpy as np

DATA_DIR = "data/ch2_6"
OUT_DIR = "reports/ch2_6/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def num_sci(x: float) -> str:
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n = np.load(os.path.join(DATA_DIR, "summary__n.npy"))
    en = np.load(os.path.join(DATA_DIR, "summary__en.npy"))

    lines = []
    lines.append("% Auto-generated from data/ch2_6/*.npy")
    lines.append("% Requires \\usepackage{booktabs}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Maximum residual component for the tridiagonal solves.}")
    lines.append(r"\label{tab:residuals}")
    lines.append(r"\begin{tabular}{rr}")
    lines.append(r"\toprule")
    lines.append(r"$n$ & $e_n = \max_i |r_i|$ \\")
    lines.append(r"\midrule")

    for n_i, e_i in zip(n, en):
        lines.append(f"{int(round(float(n_i)))} & {num_sci(float(e_i))} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
