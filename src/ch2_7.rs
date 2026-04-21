use crate::util;
use std::io;

const PART1_N: usize = 5;
const CONVERGENCE_N_VALUES: [usize; 6] = [5, 10, 20, 40, 80, 160];
const FINE_STEPS: usize = 1000;
const RATE_TOL: f64 = 1.0e-12;

type ScalarFn = fn(f64) -> f64;

#[derive(Debug, Clone, Copy)]
pub struct PointRow {
    pub x: f64,
    pub approx: f64,
    pub exact: f64,
    pub err: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SummaryRow {
    pub n: usize,
    pub h: f64,
    pub err: f64,
    pub rate: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
enum Scheme {
    ReactionDiffusion,
    AdvectionReactionBackward,
}

#[derive(Clone, Copy)]
struct Case {
    slug: &'static str,
    title: &'static str,
    exact: ScalarFn,
    rhs: ScalarFn,
    scheme: Scheme,
}

fn exact_exp(x: f64) -> f64 {
    x * (1.0 - x) * (-x).exp()
}

fn rhs_exp(x: f64) -> f64 {
    4.0 * (-x).exp() - 4.0 * x * (-x).exp()
}

fn exact_poly(x: f64) -> f64 {
    x * (1.0 - x)
}

fn rhs_poly(x: f64) -> f64 {
    2.0 + x - x.powi(2)
}

fn exact_grad(x: f64) -> f64 {
    x * (1.0 - x) * (x - 3.0)
}

fn rhs_grad(x: f64) -> f64 {
    -(x - 1.0) * (x.powi(2) - 11.0)
}

fn convergence_cases() -> [Case; 3] {
    [
        Case {
            slug: "part2",
            title: "Part 2",
            exact: exact_exp,
            rhs: rhs_exp,
            scheme: Scheme::ReactionDiffusion,
        },
        Case {
            slug: "part3",
            title: "Part 3",
            exact: exact_poly,
            rhs: rhs_poly,
            scheme: Scheme::ReactionDiffusion,
        },
        Case {
            slug: "part4",
            title: "Part 4",
            exact: exact_grad,
            rhs: rhs_grad,
            scheme: Scheme::AdvectionReactionBackward,
        },
    ]
}

fn observed_rate(prev_err: f64, err: f64) -> Option<f64> {
    if prev_err.abs() < RATE_TOL || err.abs() < RATE_TOL {
        None
    } else {
        Some((prev_err.abs() / err.abs()).ln() / 2.0_f64.ln())
    }
}

fn build_system(
    n: usize,
    rhs: ScalarFn,
    scheme: Scheme,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(n >= 2, "n must be at least 2");

    let m = n - 1;
    let h = 1.0 / n as f64;

    let mut lower = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut load = vec![0.0; m];

    for i in 0..m {
        let x = (i + 1) as f64 * h;
        load[i] = h * h * rhs(x);

        match scheme {
            Scheme::ReactionDiffusion => {
                if i > 0 {
                    lower[i] = -1.0;
                }
                diag[i] = 2.0 + h * h;
                if i + 1 < m {
                    upper[i] = -1.0;
                }
            }
            Scheme::AdvectionReactionBackward => {
                if i > 0 {
                    lower[i] = -(1.0 + h);
                }
                diag[i] = 2.0 + h + h * h;
                if i + 1 < m {
                    upper[i] = -1.0;
                }
            }
        }
    }

    (lower, diag, upper, load)
}

fn solve_tridiagonal(
    lower: &[f64],
    mut diag: Vec<f64>,
    upper: &[f64],
    mut rhs: Vec<f64>,
) -> Vec<f64> {
    let n = diag.len();
    assert!(n > 0, "system must be non-empty");
    assert_eq!(lower.len(), n, "lower diagonal length mismatch");
    assert_eq!(upper.len(), n, "upper diagonal length mismatch");
    assert_eq!(rhs.len(), n, "rhs length mismatch");

    for i in 1..n {
        let multiplier = lower[i] / diag[i - 1];
        diag[i] -= multiplier * upper[i - 1];
        rhs[i] -= multiplier * rhs[i - 1];
    }

    let mut x = vec![0.0; n];
    x[n - 1] = rhs[n - 1] / diag[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
    }

    x
}

fn solve_case(n: usize, rhs: ScalarFn, scheme: Scheme) -> Vec<(f64, f64)> {
    let h = 1.0 / n as f64;
    let (lower, diag, upper, load) = build_system(n, rhs, scheme);
    let interior = solve_tridiagonal(&lower, diag, &upper, load);

    let mut values = Vec::with_capacity(n + 1);
    values.push((0.0, 0.0));
    for (i, ui) in interior.into_iter().enumerate() {
        values.push(((i + 1) as f64 * h, ui));
    }
    values.push((1.0, 0.0));
    values
}

fn max_error(solution: &[(f64, f64)], exact: ScalarFn) -> f64 {
    solution
        .iter()
        .copied()
        .map(|(x, approx)| (exact(x) - approx).abs())
        .fold(0.0, f64::max)
}

fn fine_grid() -> Vec<f64> {
    (0..=FINE_STEPS)
        .map(|i| i as f64 / FINE_STEPS as f64)
        .collect()
}

pub fn part1_rows() -> Vec<PointRow> {
    solve_case(PART1_N, rhs_exp, Scheme::ReactionDiffusion)
        .into_iter()
        .skip(1)
        .take(PART1_N - 1)
        .map(|(x, approx)| {
            let exact = exact_exp(x);
            PointRow {
                x,
                approx,
                exact,
                err: exact - approx,
            }
        })
        .collect()
}

fn summarize(case: Case) -> Vec<SummaryRow> {
    CONVERGENCE_N_VALUES
        .iter()
        .copied()
        .map(|n| {
            let solution = solve_case(n, case.rhs, case.scheme);
            let h = 1.0 / n as f64;
            let err = max_error(&solution, case.exact);
            (n, h, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (n, h, err)| {
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow { n, h, err, rate })
        })
        .collect()
}

pub fn part2_summary() -> Vec<SummaryRow> {
    summarize(convergence_cases()[0])
}

pub fn part3_summary() -> Vec<SummaryRow> {
    summarize(convergence_cases()[1])
}

pub fn part4_summary() -> Vec<SummaryRow> {
    summarize(convergence_cases()[2])
}

fn write_part1_data(rows: &[PointRow]) {
    let out_dir = String::from("data/ch2_7");
    let x: Vec<f64> = rows.iter().map(|row| row.x).collect();
    let approx: Vec<f64> = rows.iter().map(|row| row.approx).collect();
    let exact: Vec<f64> = rows.iter().map(|row| row.exact).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();

    util::write_data(&x, out_dir.clone(), String::from("part1__x"));
    util::write_data(&approx, out_dir.clone(), String::from("part1__approx"));
    util::write_data(&exact, out_dir.clone(), String::from("part1__exact"));
    util::write_data(&err, out_dir, String::from("part1__err"));
}

fn write_plot_data() {
    let out_dir = String::from("data/ch2_7");
    let nodes = solve_case(PART1_N, rhs_exp, Scheme::ReactionDiffusion);
    let nodes_x: Vec<f64> = nodes.iter().map(|(x, _)| *x).collect();
    let nodes_y: Vec<f64> = nodes.iter().map(|(_, y)| *y).collect();

    let x = fine_grid();
    let exact: Vec<f64> = x.iter().copied().map(exact_exp).collect();

    util::write_data(&x, out_dir.clone(), String::from("plot__x"));
    util::write_data(&exact, out_dir.clone(), String::from("plot__exact"));
    util::write_data(&nodes_x, out_dir.clone(), String::from("plot__nodes_x"));
    util::write_data(&nodes_y, out_dir, String::from("plot__nodes_y"));
}

fn write_summary_data(name: &str, rows: &[SummaryRow]) {
    let out_dir = String::from("data/ch2_7");
    let n: Vec<f64> = rows.iter().map(|row| row.n as f64).collect();
    let h: Vec<f64> = rows.iter().map(|row| row.h).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();
    let rate: Vec<f64> = rows
        .iter()
        .map(|row| row.rate.unwrap_or(f64::NAN))
        .collect();

    util::write_data(&n, out_dir.clone(), format!("{name}__n"));
    util::write_data(&h, out_dir.clone(), format!("{name}__h"));
    util::write_data(&err, out_dir.clone(), format!("{name}__err"));
    util::write_data(&rate, out_dir, format!("{name}__rate"));
}

pub fn generate() -> io::Result<()> {
    let part1 = part1_rows();
    let summaries = [
        (convergence_cases()[0], part2_summary()),
        (convergence_cases()[1], part3_summary()),
        (convergence_cases()[2], part4_summary()),
    ];

    for row in &part1 {
        println!(
            "x = {:.1}, U_k = {:.10e}, u(x_k) = {:.10e}, error = {:.10e}",
            row.x, row.approx, row.exact, row.err
        );
    }

    for (case, rows) in &summaries {
        println!("{}", case.title);
        for row in rows {
            println!(
                "  N = {:>3}, h = {:.5}, E_h = {:.10e}, rate = {}",
                row.n,
                row.h,
                row.err,
                row.rate
                    .map(|rate| format!("{rate:.6}"))
                    .unwrap_or_else(|| String::from("--"))
            );
        }
    }

    write_part1_data(&part1);
    write_plot_data();
    for (case, rows) in &summaries {
        write_summary_data(case.slug, rows);
    }

    util::plot("ch2_7")?;
    util::run_python_script("scripts/ch2_7/make_tables.py")?;
    util::copy_file(
        "plots/ch2_7/approximation.png",
        "reports/ch2_7/figures/approximation.png",
    )?;
    util::copy_file("plots/ch2_7/error.png", "reports/ch2_7/figures/error.png")?;
    util::build_report("reports/ch2_7", "2.7.pdf")
}

#[cfg(test)]
mod tests {
    use super::{PART1_N, part1_rows, part2_summary, part3_summary, part4_summary};

    #[test]
    fn part1_has_expected_mesh_points() {
        let rows = part1_rows();

        assert_eq!(rows.len(), PART1_N - 1);
        assert!((rows[0].x - 0.2).abs() < 1.0e-12);
        assert!((rows[3].x - 0.8).abs() < 1.0e-12);
    }

    #[test]
    fn part2_converges_second_order() {
        let rows = part2_summary();
        let last = rows.last().expect("missing Part 2 rows");

        assert!(rows[1].err < rows[0].err);
        assert!((last.rate.expect("missing Part 2 rate") - 2.0).abs() < 0.06);
    }

    #[test]
    fn part3_is_exact_to_roundoff() {
        let rows = part3_summary();

        for row in rows {
            assert!(row.err < 1.0e-12, "Part 3 error too large for N={}", row.n);
        }
    }

    #[test]
    fn part4_converges_first_order() {
        let rows = part4_summary();
        let last = rows.last().expect("missing Part 4 rows");

        assert!(rows[1].err < rows[0].err);
        assert!((last.rate.expect("missing Part 4 rate") - 1.0).abs() < 0.06);
    }
}
