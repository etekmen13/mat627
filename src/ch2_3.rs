use crate::util;
use std::io;

const T0: f64 = 1.0;
const T1: f64 = 2.0;
const Y0: f64 = 2.0;

type StepFn = fn(f64, f64, f64) -> f64;

#[derive(Debug, Clone, Copy)]
pub struct StepRow {
    pub k: usize,
    pub t: f64,
    pub approx: f64,
    pub exact: f64,
    pub err: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SummaryRow {
    pub h: f64,
    pub approx: f64,
    pub err: f64,
    pub rate: Option<f64>,
}

fn rhs(t: f64, y: f64) -> f64 {
    (2.0 * t).sin() / t.powi(2) - 2.0 * y / t
}

fn exact(t: f64) -> f64 {
    (4.0 + 2.0_f64.cos() - (2.0 * t).cos()) / (2.0 * t.powi(2))
}

fn euler_step(t: f64, y: f64, h: f64) -> f64 {
    y + h * rhs(t, y)
}

fn rk4_step(t: f64, y: f64, h: f64) -> f64 {
    let k1 = rhs(t, y);
    let k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1);
    let k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2);
    let k4 = rhs(t + h, y + h * k3);

    y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}

fn step_count(start: f64, end: f64, h: f64) -> usize {
    let n = (end - start) / h;
    let rounded = n.round();
    assert!((n - rounded).abs() < 1.0e-12);
    rounded as usize
}

fn solve(step: StepFn, h: f64) -> Vec<(f64, f64)> {
    let n = step_count(T0, T1, h);
    let mut rows = Vec::with_capacity(n + 1);
    let mut t = T0;
    let mut y = Y0;

    rows.push((t, y));
    for k in 0..n {
        y = step(t, y, h);
        t = T0 + (k + 1) as f64 * h;
        rows.push((t, y));
    }

    rows
}

fn observed_rate(prev_err: f64, err: f64) -> Option<f64> {
    let a = prev_err.abs();
    let b = err.abs();

    if a == 0.0 || b == 0.0 {
        None
    } else {
        Some((a / b).ln() / 2.0_f64.ln())
    }
}

pub fn euler_part1() -> Vec<StepRow> {
    solve(euler_step, 0.25)
        .into_iter()
        .enumerate()
        .map(|(k, (t, approx))| {
            let exact = exact(t);

            StepRow {
                k,
                t,
                approx,
                exact,
                err: exact - approx,
            }
        })
        .collect()
}

fn summarize(step: StepFn, hs: &[f64]) -> Vec<SummaryRow> {
    let exact_end = exact(T1);

    hs.iter()
        .copied()
        .map(|h| {
            let approx = solve(step, h)
                .last()
                .map(|(_, y)| *y)
                .expect("solver returned no steps");
            let err = exact_end - approx;
            (h, approx, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (h, approx, err)| {
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow {
                h,
                approx,
                err,
                rate,
            })
        })
        .collect()
}

pub fn euler_summary() -> Vec<SummaryRow> {
    summarize(euler_step, &[0.5, 0.25, 0.125, 0.0625, 0.03125])
}

pub fn rk4_summary() -> Vec<SummaryRow> {
    summarize(rk4_step, &[0.5, 0.25, 0.125, 0.0625, 0.03125])
}

fn extrapolated_euler(h: f64) -> f64 {
    let n = step_count(T0, T1, 2.0 * h);
    let mut t = T0;
    let mut yr = Y0;

    for _ in 0..n {
        let z1 = euler_step(t, yr, h);
        let z2 = euler_step(t + h, z1, h);
        let z_bar = euler_step(t, yr, 2.0 * h);

        yr = 2.0 * z2 - z_bar;
        t += 2.0 * h;
    }

    yr
}

pub fn extrapolated_summary() -> Vec<SummaryRow> {
    let hs = [0.25, 0.125, 0.0625, 0.03125];
    let exact_end = exact(T1);

    hs.into_iter()
        .map(|h| {
            let approx = extrapolated_euler(h);
            let err = exact_end - approx;
            (h, approx, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (h, approx, err)| {
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow {
                h,
                approx,
                err,
                rate,
            })
        })
        .collect()
}

pub fn generate() -> io::Result<()> {
    let part1 = euler_part1();
    let euler = euler_summary();
    let rk4 = rk4_summary();
    let extrap = extrapolated_summary();

    write_part1_data(&part1);
    write_summary_data("euler", &euler);
    write_summary_data("rk4", &rk4);
    write_summary_data("extrapolated", &extrap);
    write_plot_data(&part1);

    util::plot("ch2_3")?;
    util::run_python_script("scripts/ch2_3/make_tables.py")?;
    util::copy_file("plots/ch2_3/plot.png", "reports/ch2_3/figures/plot.png")?;
    util::build_report("reports/ch2_3", "2.3.pdf")
}

fn write_part1_data(rows: &[StepRow]) {
    let out_dir = String::from("data/ch2_3");
    let ks: Vec<f64> = rows.iter().map(|row| row.k as f64).collect();
    let t: Vec<f64> = rows.iter().map(|row| row.t).collect();
    let approx: Vec<f64> = rows.iter().map(|row| row.approx).collect();
    let exact_vals: Vec<f64> = rows.iter().map(|row| row.exact).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();

    util::write_data(&ks, out_dir.clone(), String::from("part1__k"));
    util::write_data(&t, out_dir.clone(), String::from("part1__t"));
    util::write_data(&approx, out_dir.clone(), String::from("part1__approx"));
    util::write_data(&exact_vals, out_dir.clone(), String::from("part1__exact"));
    util::write_data(&err, out_dir, String::from("part1__err"));
}

fn write_summary_data(name: &str, rows: &[SummaryRow]) {
    let out_dir = String::from("data/ch2_3");
    let h: Vec<f64> = rows.iter().map(|row| row.h).collect();
    let approx: Vec<f64> = rows.iter().map(|row| row.approx).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();
    let rate: Vec<f64> = rows
        .iter()
        .map(|row| row.rate.unwrap_or(f64::NAN))
        .collect();

    util::write_data(&h, out_dir.clone(), format!("{name}__h"));
    util::write_data(&approx, out_dir.clone(), format!("{name}__approx"));
    util::write_data(&err, out_dir.clone(), format!("{name}__err"));
    util::write_data(&rate, out_dir, format!("{name}__rate"));
}

fn write_plot_data(rows: &[StepRow]) {
    let out_dir = String::from("data/ch2_3");
    let t_euler: Vec<f64> = rows.iter().map(|row| row.t).collect();
    let y_euler: Vec<f64> = rows.iter().map(|row| row.approx).collect();

    let t_exact: Vec<f64> = (0..=400)
        .map(|i| T0 + (T1 - T0) * (i as f64) / 400.0)
        .collect();
    let y_exact: Vec<f64> = t_exact.iter().copied().map(exact).collect();

    util::write_data(&t_exact, out_dir.clone(), String::from("plot__exact_t"));
    util::write_data(&y_exact, out_dir.clone(), String::from("plot__exact_y"));
    util::write_data(&t_euler, out_dir.clone(), String::from("plot__euler_t"));
    util::write_data(&y_euler, out_dir, String::from("plot__euler_y"));
}

#[cfg(test)]
mod tests {
    use super::{Y0, euler_summary, exact, extrapolated_summary, rk4_summary};

    #[test]
    fn exact_solution_matches_initial_value() {
        assert!((exact(1.0) - Y0).abs() < 1.0e-12);
    }

    #[test]
    fn euler_converges_first_order() {
        let rows = euler_summary();
        let last = rows.last().expect("missing euler rows");

        assert!(rows[1].err.abs() < rows[0].err.abs());
        assert!((last.rate.expect("missing euler rate") - 1.0).abs() < 0.1);
    }

    #[test]
    fn rk4_converges_fourth_order() {
        let rows = rk4_summary();
        let last = rows.last().expect("missing rk4 rows");

        assert!(rows[1].err.abs() < rows[0].err.abs());
        assert!((last.rate.expect("missing rk4 rate") - 4.0).abs() < 0.15);
    }

    #[test]
    fn extrapolation_improves_euler_order() {
        let rows = extrapolated_summary();
        let last = rows.last().expect("missing extrapolation rows");

        assert!(rows[1].err.abs() < rows[0].err.abs());
        assert!((last.rate.expect("missing extrapolation rate") - 2.0).abs() < 0.15);
    }
}
