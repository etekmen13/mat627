import os
import re
import glob
from collections import defaultdict

import numpy as np

DATA_DIR = "data/ch2_2"
OUT_DIR = "reports/ch2_2/figures"
OUT_FILE = os.path.join(OUT_DIR, "fd_tables.tex")


def pretty_case_slug(case_slug: str) -> str:
    mapping = {
        "sqrt_x_1": r"$\sqrt{x+1}$",
        "sqrt_x_plus_1": r"$\sqrt{x+1}$",
        "exp_x": r"$e^x$",
    }
    return mapping.get(case_slug, case_slug.replace("_", r"\_"))


def pretty_method_slug(method_slug: str) -> str:
    mapping = {
        "forward": "Forward difference",
        "backward": "Backward difference",
        "center": "Centered difference",
        "special": "Richardson-extrapolated forward difference",
    }
    return mapping.get(method_slug, method_slug.replace("_", " ").title())


# Column headers you requested
APPROX_HDR = {
    "forward": r"$D_h^+ f(1)$",
    "backward": r"$D_h^- f(1)$",
    "center": r"$D_h f(1)$",
    "special": r"$\widetilde D_h^+ f(1)$",
}

ERR_HDR = {
    "forward": r"$E_h = f'(1) - D_h^+ f(1)$",
    "backward": r"$E_h = f'(1) - D_h^- f(1)$",
    "center": r"$E_h = f'(1) - D_h f(1)$",
    "special": r"$E_h = f'(1) - \widetilde D_h^+ f(1)$",
}

RATE_HDR = r"$\frac{\ln\left|\frac{E_{2h}}{E_h}\right|}{\ln 2}$"


def load_data():
    data = defaultdict(lambda: defaultdict(dict))
    exact = {}

    exact_pat = re.compile(r"^(?P<case>.+)__exact\.npy$")
    arr_pat = re.compile(
        r"^(?P<case>.+)__(?P<method>.+)__(?P<kind>h|approx|err|abs_err|order)\.npy$"
    )

    for path in glob.glob(os.path.join(DATA_DIR, "*.npy")):
        fname = os.path.basename(path)

        m = exact_pat.match(fname)
        if m:
            case = m.group("case")
            exact[case] = float(np.load(path)[0])
            continue

        m = arr_pat.match(fname)
        if m:
            case = m.group("case")
            method = m.group("method")
            kind = m.group("kind")
            data[case][method][kind] = np.load(path)

    return data, exact


def num_sci(x: float) -> str:
    """
    Plain LaTeX scientific notation, no siunitx required.
    Produces: $-1.234567\\times 10^{-3}$
    """
    s = f"{x:.6e}"
    mant, exp = s.split("e")
    exp_i = int(exp)
    return rf"${mant}\times 10^{{{exp_i}}}$"


def num_hinv(h: float) -> str:
    inv = 1.0 / h
    r = round(inv)
    if abs(inv - r) < 1e-12:
        return rf"${int(r)}$"
    return num_sci(inv)


def make_table(case_slug, method_slug, d):
    h = d["h"]
    approx = d["approx"]
    err = d["err"]
    order = d.get("order", np.full_like(h, np.nan))

    approx_hdr = APPROX_HDR.get(method_slug, method_slug.replace("_", r"\_"))
    err_hdr = ERR_HDR.get(method_slug, method_slug.replace("_", r"\_"))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(
        rf"\caption{{{pretty_method_slug(method_slug)} for {pretty_case_slug(case_slug)}.}}"
    )
    lines.append(rf"\label{{tab:{case_slug}_{method_slug}}}")

    # Force-fit to page width (robust against overflow)
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrrr}")
    lines.append(r"\toprule")
    lines.append(rf"$h^{{-1}}$ & {approx_hdr} & {err_hdr} & {RATE_HDR} \\")
    lines.append(r"\midrule")

    for i in range(len(h)):
        rate_str = r"--" if not np.isfinite(order[i]) else num_sci(float(order[i]))
        row = (
            f"{num_hinv(float(h[i]))} & "
            f"{num_sci(float(approx[i]))} & "
            f"{num_sci(float(err[i]))} & "
            f"{rate_str} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")  # end resizebox
    lines.append(r"\end{table}")
    lines.append("")  # blank line
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, exact = load_data()

    out = []
    out.append("% Auto-generated from data/ch2_2/*.npy")
    out.append("% Requires \\usepackage{booktabs}")
    out.append("% Requires \\usepackage{graphicx} (for \\resizebox)")
    out.append(
        "% Tables are forced to fit \\textwidth via \\resizebox{\\textwidth}{!}{...}"
    )
    out.append("")

    for case_slug, methods in sorted(data.items()):
        out.append(rf"\section{{{pretty_case_slug(case_slug)}}}")
        out.append("")

        for method_slug, d in sorted(methods.items()):
            out.append(rf"\subsection{{{pretty_method_slug(method_slug)}}}")
            out.append("")
            out.append(make_table(case_slug, method_slug, d))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote LaTeX tables to {OUT_FILE}")


if __name__ == "__main__":
    main()
