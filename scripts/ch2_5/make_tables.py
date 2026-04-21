import os

import numpy as np

DATA_DIR = "data/ch2_5"
OUT_DIR = "reports/ch2_5/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "approx": np.load(os.path.join(DATA_DIR, f"{name}__approx.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def num_sci(x: float) -> str:
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def num_h(x: float) -> str:
    return f"{float(x):.5f}".rstrip("0").rstrip(".")


def num_rate(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{float(x):.6f}"


def make_table(title: str, label: str, interval_tex: str, func_tex: str, summary):
    lines = []
    lines.append(rf"\section{{{title}}}")
    lines.append(rf"\noindent Composite trapezoid rule for {func_tex} on {interval_tex}.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{Composite trapezoid-rule results on {interval_tex}.}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{rrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $h$ & $T_N(f)$ & $E_h = |T_N(f)-I(f)|$ & Rate \\")
    lines.append(r"\midrule")

    for n, h, approx, err, rate in zip(
        summary["n"],
        summary["h"],
        summary["approx"],
        summary["err"],
        summary["rate"],
    ):
        lines.append(
            f"{int(round(float(n)))} & {num_h(h)} & {num_sci(float(approx))} & "
            f"{num_sci(float(err))} & {num_rate(float(rate))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out = []
    out.append("% Auto-generated from data/ch2_5/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("")
    out.append(
        make_table("Part 1", "part1", r"$[1,3]$", r"$f(x)=x^2e^{-x}$", load_summary("part1"))
    )
    out.append(
        make_table("Part 2", "part2", r"$[0,2]$", r"$f(x)=x^2e^{-x}$", load_summary("part2"))
    )
    out.append(
        make_table("Part 3", "part3", r"$[0,1]$", r"$f(x)=\sqrt{x}$", load_summary("part3"))
    )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
