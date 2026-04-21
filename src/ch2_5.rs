use crate::util;
use std::io;

const SMOOTH_N_VALUES: [usize; 5] = [2, 4, 8, 16, 32];
const SQRT_N_VALUES: [usize; 6] = [1, 2, 4, 8, 16, 32];
const PLOT_N: usize = 4;
const PLOT_STEPS: usize = 400;

type ScalarFn = fn(f64) -> f64;

#[derive(Debug, Clone, Copy)]
pub struct SummaryRow {
    pub n: usize,
    pub h: f64,
    pub approx: f64,
    pub err: f64,
    pub rate: Option<f64>,
}

#[derive(Clone, Copy)]
struct Case {
    slug: &'static str,
    title: &'static str,
    a: f64,
    b: f64,
    ns: &'static [usize],
    exact: f64,
    f: ScalarFn,
}

fn smooth_integrand(x: f64) -> f64 {
    x.powi(2) * (-x).exp()
}

fn sqrt_integrand(x: f64) -> f64 {
    x.sqrt()
}

fn smooth_exact_integral(a: f64, b: f64) -> f64 {
    antiderivative(b) - antiderivative(a)
}

fn antiderivative(x: f64) -> f64 {
    -(x.powi(2) + 2.0 * x + 2.0) * (-x).exp()
}

fn cases() -> [Case; 3] {
    [
        Case {
            slug: "part1",
            title: "Part 1",
            a: 1.0,
            b: 3.0,
            ns: &SMOOTH_N_VALUES,
            exact: smooth_exact_integral(1.0, 3.0),
            f: smooth_integrand,
        },
        Case {
            slug: "part2",
            title: "Part 2",
            a: 0.0,
            b: 2.0,
            ns: &SMOOTH_N_VALUES,
            exact: smooth_exact_integral(0.0, 2.0),
            f: smooth_integrand,
        },
        Case {
            slug: "part3",
            title: "Part 3",
            a: 0.0,
            b: 1.0,
            ns: &SQRT_N_VALUES,
            exact: 2.0 / 3.0,
            f: sqrt_integrand,
        },
    ]
}

fn composite_trapezoid(f: ScalarFn, a: f64, b: f64, n: usize) -> f64 {
    assert!(n > 0, "n must be positive");

    let h = (b - a) / n as f64;
    let interior_sum: f64 = (1..n).map(|i| f(a + i as f64 * h)).sum();

    0.5 * h * (f(a) + 2.0 * interior_sum + f(b))
}

fn observed_rate(prev_err: f64, err: f64) -> Option<f64> {
    if prev_err == 0.0 || err == 0.0 {
        None
    } else {
        Some((prev_err / err).ln() / 2.0_f64.ln())
    }
}

fn summarize(case: Case) -> Vec<SummaryRow> {
    case.ns
        .iter()
        .copied()
        .map(|n| {
            let h = (case.b - case.a) / n as f64;
            let approx = composite_trapezoid(case.f, case.a, case.b, n);
            let err = (approx - case.exact).abs();
            (n, h, approx, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (n, h, approx, err)| {
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow {
                n,
                h,
                approx,
                err,
                rate,
            })
        })
        .collect()
}

fn fine_grid(a: f64, b: f64) -> Vec<f64> {
    (0..=PLOT_STEPS)
        .map(|i| a + (b - a) * i as f64 / PLOT_STEPS as f64)
        .collect()
}

fn interval_index(a: f64, b: f64, n: usize, x: f64) -> usize {
    if (x - b).abs() < 1.0e-12 {
        return n - 1;
    }

    let h = (b - a) / n as f64;
    (((x - a) / h).floor() as usize).min(n - 1)
}

fn piecewise_linear_values(f: ScalarFn, a: f64, b: f64, n: usize, xs: &[f64]) -> Vec<f64> {
    let h = (b - a) / n as f64;

    xs.iter()
        .copied()
        .map(|x| {
            let i = interval_index(a, b, n, x);
            let x_left = a + i as f64 * h;
            let x_right = x_left + h;
            let f_left = f(x_left);
            let f_right = f(x_right);

            f_left * (x_right - x) / h + f_right * (x - x_left) / h
        })
        .collect()
}

fn write_summary_data(case: Case, rows: &[SummaryRow]) {
    let out_dir = String::from("data/ch2_5");
    let n: Vec<f64> = rows.iter().map(|row| row.n as f64).collect();
    let h: Vec<f64> = rows.iter().map(|row| row.h).collect();
    let approx: Vec<f64> = rows.iter().map(|row| row.approx).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();
    let rate: Vec<f64> = rows
        .iter()
        .map(|row| row.rate.unwrap_or(f64::NAN))
        .collect();

    util::write_data(&n, out_dir.clone(), format!("{}__n", case.slug));
    util::write_data(&h, out_dir.clone(), format!("{}__h", case.slug));
    util::write_data(&approx, out_dir.clone(), format!("{}__approx", case.slug));
    util::write_data(&err, out_dir.clone(), format!("{}__err", case.slug));
    util::write_data(&rate, out_dir, format!("{}__rate", case.slug));
}

fn write_plot_data(case: Case) {
    let out_dir = String::from("data/ch2_5");
    let x = fine_grid(case.a, case.b);
    let exact: Vec<f64> = x.iter().copied().map(case.f).collect();
    let interp = piecewise_linear_values(case.f, case.a, case.b, PLOT_N, &x);
    let nodes_x: Vec<f64> = (0..=PLOT_N)
        .map(|i| case.a + (case.b - case.a) * i as f64 / PLOT_N as f64)
        .collect();
    let nodes_y: Vec<f64> = nodes_x.iter().copied().map(case.f).collect();

    util::write_data(&x, out_dir.clone(), format!("plot__{}__x", case.slug));
    util::write_data(
        &exact,
        out_dir.clone(),
        format!("plot__{}__exact", case.slug),
    );
    util::write_data(
        &interp,
        out_dir.clone(),
        format!("plot__{}__interp", case.slug),
    );
    util::write_data(
        &nodes_x,
        out_dir.clone(),
        format!("plot__{}__nodes_x", case.slug),
    );
    util::write_data(&nodes_y, out_dir, format!("plot__{}__nodes_y", case.slug));
}

pub fn generate() -> io::Result<()> {
    for case in cases() {
        let rows = summarize(case);
        println!("{}: exact integral = {:.10e}", case.title, case.exact);
        write_summary_data(case, &rows);
        write_plot_data(case);
    }

    util::plot("ch2_5")?;
    util::run_python_script("scripts/ch2_5/make_tables.py")?;
    util::copy_file(
        "plots/ch2_5/trapezoids.png",
        "reports/ch2_5/figures/trapezoids.png",
    )?;
    util::copy_file("plots/ch2_5/error.png", "reports/ch2_5/figures/error.png")?;
    util::build_report("reports/ch2_5", "2.5.pdf")
}

#[cfg(test)]
mod tests {
    use super::{cases, composite_trapezoid, smooth_exact_integral, summarize};

    #[test]
    fn smooth_exact_integrals_match_assignment_values() {
        let part1 = smooth_exact_integral(1.0, 3.0);
        let part2 = smooth_exact_integral(0.0, 2.0);

        assert!((part1 - 0.9930170436).abs() < 1.0e-10);
        assert!((part2 - 0.6466471676).abs() < 1.0e-10);
    }

    #[test]
    fn trapezoid_rule_is_exact_for_linear_functions() {
        let approx = composite_trapezoid(|x| 3.0 * x - 1.0, 0.0, 2.0, 8);
        let exact = 4.0;

        assert!((approx - exact).abs() < 1.0e-12);
    }

    #[test]
    fn smooth_cases_converge_quadratically() {
        let [part1, part2, _] = cases();
        let rows1 = summarize(part1);
        let rows2 = summarize(part2);
        let rate1 = rows1.last().expect("missing part1 rows").rate.unwrap();
        let rate2 = rows2.last().expect("missing part2 rows").rate.unwrap();

        assert!((rate1 - 2.0).abs() < 0.05);
        assert!((rate2 - 4.0).abs() < 0.05);
    }

    #[test]
    fn sqrt_case_converges_like_h_to_three_halves() {
        let [_, _, part3] = cases();
        let rows = summarize(part3);
        let rate = rows.last().expect("missing part3 rows").rate.unwrap();

        assert!((rate - 1.5).abs() < 0.08);
    }
}
