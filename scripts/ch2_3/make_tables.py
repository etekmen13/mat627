import os

import numpy as np

DATA_DIR = "data/ch2_3"
OUT_DIR = "reports/ch2_3/figures"
OUT_FILE = os.path.join(OUT_DIR, "tables.tex")


def num_sci(x: float) -> str:
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def num_rate(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{float(x):.6f}"


def num_h(x: float) -> str:
    return f"{float(x):.5f}".rstrip("0").rstrip(".")


def num_t(x: float) -> str:
    return f"{float(x):.2f}"


def load_part1():
    return {
        "k": np.load(os.path.join(DATA_DIR, "part1__k.npy")),
        "t": np.load(os.path.join(DATA_DIR, "part1__t.npy")),
        "approx": np.load(os.path.join(DATA_DIR, "part1__approx.npy")),
        "exact": np.load(os.path.join(DATA_DIR, "part1__exact.npy")),
        "err": np.load(os.path.join(DATA_DIR, "part1__err.npy")),
    }


def load_summary(name: str):
    return {
        "h": np.load(os.path.join(DATA_DIR, f"{name}__h.npy")),
        "approx": np.load(os.path.join(DATA_DIR, f"{name}__approx.npy")),
        "err": np.load(os.path.join(DATA_DIR, f"{name}__err.npy")),
        "rate": np.load(os.path.join(DATA_DIR, f"{name}__rate.npy")),
    }


def make_part1_table(part1):
    lines = []
    lines.append(r"\section{Part 1}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\caption{Euler's method with $h=\tfrac{1}{4}$.}")
    lines.append(r"\label{tab:part1}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"$k$ & $t_k$ & $y_k$ & $y(t_k)$ & Error $= y(t_k) - y_k$ \\"
    )
    lines.append(r"\midrule")

    for k, t, approx, exact, err in zip(
        part1["k"], part1["t"], part1["approx"], part1["exact"], part1["err"]
    ):
        lines.append(
            f"{int(round(float(k)))} & {num_t(t)} & {num_sci(float(approx))} & "
            f"{num_sci(float(exact))} & {num_sci(float(err))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_summary_table(
    title: str, label: str, summary, approx_header: str, err_header: str
):
    lines = []
    lines.append(rf"\section{{{title}}}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{{title}.}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(rf"$h$ & {approx_header} & {err_header} & Rate \\")
    lines.append(r"\midrule")

    for h, approx, err, rate in zip(
        summary["h"], summary["approx"], summary["err"], summary["rate"]
    ):
        lines.append(
            f"{num_h(h)} & {num_sci(float(approx))} & {num_sci(float(err))} & "
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
    out.append("% Auto-generated from data/ch2_3/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("% Requires \\usepackage{graphicx}")
    out.append("")
    out.append(make_part1_table(load_part1()))
    out.append(
        make_summary_table(
            "Part 3",
            "part3",
            load_summary("euler"),
            r"$y_N$",
            r"$E_h = y(2) - y_N$",
        )
    )
    out.append(
        make_summary_table(
            "Part 4",
            "part4",
            load_summary("rk4"),
            r"$y_N$",
            r"$E_h = y(2) - y_N$",
        )
    )
    out.append(
        make_summary_table(
            "Part 5",
            "part5",
            load_summary("extrapolated"),
            r"$y_N^r$",
            r"$E_h = y(2) - y_N^r$",
        )
    )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
