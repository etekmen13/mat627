import os

import numpy as np

DATA_DIR = "data/ch2_4"
OUT_DIR = "reports/ch2_4/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def load_summary(name: str):
    return {
        "n": np.load(os.path.join(DATA_DIR, f"{name}__n.npy")),
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
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


def make_table(title: str, label: str, interval_tex: str, summary):
    lines = []
    lines.append(rf"\section{{{title}}}")
    lines.append(rf"\noindent Uniform piecewise linear interpolation on {interval_tex}.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        rf"\caption{{Maximum fine-grid error for $f(x)=x^{{1/3}}$ on {interval_tex}.}}"
    )
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $h$ & $E_h = \max_{z_i \in D}|q_N(z_i) - f(z_i)|$ & Rate \\")
    lines.append(r"\midrule")

    for n, h, err, rate in zip(
        summary["n"], summary["h"], summary["err"], summary["rate"]
    ):
        lines.append(
            f"{int(round(float(n)))} & {num_h(h)} & {num_sci(float(err))} & "
            f"{num_rate(float(rate))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    smooth = load_summary("smooth")
    singular = load_summary("singular")

    out = []
    out.append("% Auto-generated from data/ch2_4/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("")
    out.append(make_table("Part 2", "smooth", r"$[1,2]$", smooth))
    out.append(make_table("Part 3", "singular", r"$[0,1]$", singular))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
