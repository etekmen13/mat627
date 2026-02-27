import os
import re
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/ch2_2"
PLOT_DIR = "plots/ch2_2"


def pretty_case(case_slug: str) -> str:
    mapping = {
        "sqrt_x_1": r"$\sqrt{x+1}$",
        "exp_x": r"$e^x$",
        "sqrt_x_plus_1": r"$\sqrt{x+1}$",
        "exp_x_": r"$e^x$",
    }
    return mapping.get(case_slug, case_slug.replace("_", " "))


def pretty_method(method_slug: str) -> str:
    mapping = {
        "forward": "forward",
        "backward": "backward",
        "center": "centered",
        "special": "special",
    }
    return mapping.get(method_slug, method_slug.replace("_", " "))


def load_data():
    os.makedirs(PLOT_DIR, exist_ok=True)

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
            exact_arr = np.load(path)
            exact[case] = float(exact_arr[0])
            continue

        m = arr_pat.match(fname)
        if m:
            case = m.group("case")
            method = m.group("method")
            kind = m.group("kind")
            data[case][method][kind] = np.load(path)

    return data, exact


def plot_case(case, methods, exact_val):
    plt.figure(figsize=(8, 5))
    for method, d in sorted(methods.items()):
        h = d["h"]
        approx = d["approx"]
        plt.plot(h, approx, marker="o", label=pretty_method(method))

    plt.axhline(exact_val, linestyle="--", label=f"exact = {exact_val:.8e}")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("Derivative approximation")
    plt.title(f"Finite-difference approximations for {pretty_case(case)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{case}_approx.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for method, d in sorted(methods.items()):
        h = d["h"]
        abs_err = d.get("abs_err", np.abs(d["err"]))
        plt.loglog(h, abs_err, marker="o", label=pretty_method(method))

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$|$error$|$")
    plt.title(f"Finite-difference error for {pretty_case(case)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{case}_error.png"), dpi=150)
    plt.close()


def main():
    data, exact = load_data()

    for case, methods in sorted(data.items()):
        if case not in exact:
            any_method = next(iter(methods))
            d = methods[any_method]
            exact_val = float(d["approx"][0] + d["err"][0])
        else:
            exact_val = exact[case]

        plot_case(case, methods, exact_val)

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
