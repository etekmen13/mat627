use crate::util;
use std::io;

const PLOT_A: f64 = 0.0;
const PLOT_B: f64 = 2.0;
const PLOT_N: usize = 4;
const FINE_STEPS: usize = 1000;
const N_VALUES: [usize; 6] = [1, 2, 4, 8, 16, 32];

#[derive(Debug, Clone, Copy)]
pub struct SummaryRow {
    pub n: usize,
    pub h: f64,
    pub err: f64,
    pub rate: Option<f64>,
}

fn f(x: f64) -> f64 {
    x.cbrt()
}

fn mesh(a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    assert!(n > 0, "n must be positive");

    let h = (b - a) / n as f64;

    (0..n)
        .map(|i| {
            let left = a + i as f64 * h;
            let right = a + (i + 1) as f64 * h;
            (left, right)
        })
        .collect()
}

fn linear_interp<F>(a: f64, b: f64, f: F) -> impl Fn(f64) -> f64
where
    F: Fn(f64) -> f64,
{
    assert!(a != b, "interval endpoints must be distinct");

    let fa = f(a);
    let fb = f(b);

    move |x| fa * (b - x) / (b - a) + fb * (x - a) / (b - a)
}

fn fine_grid(a: f64, b: f64) -> Vec<f64> {
    (0..=FINE_STEPS)
        .map(|i| a + (b - a) * i as f64 / FINE_STEPS as f64)
        .collect()
}

fn interval_index(a: f64, b: f64, n: usize, x: f64) -> usize {
    assert!(x >= a - 1.0e-12 && x <= b + 1.0e-12, "x must lie in [a, b]");

    if (x - b).abs() < 1.0e-12 {
        return n - 1;
    }

    let h = (b - a) / n as f64;
    (((x - a) / h).floor() as usize).min(n - 1)
}

fn piecewise_linear_values(a: f64, b: f64, n: usize, xs: &[f64]) -> Vec<f64> {
    let pieces: Vec<_> = mesh(a, b, n)
        .into_iter()
        .map(|(left, right)| linear_interp(left, right, f as fn(f64) -> f64))
        .collect();

    xs.iter()
        .copied()
        .map(|x| pieces[interval_index(a, b, n, x)](x))
        .collect()
}

fn observed_rate(prev_err: f64, err: f64) -> Option<f64> {
    if prev_err == 0.0 || err == 0.0 {
        None
    } else {
        Some((prev_err / err).ln() / 2.0_f64.ln())
    }
}

fn summarize_case(a: f64, b: f64) -> Vec<SummaryRow> {
    let z = fine_grid(a, b);

    N_VALUES
        .iter()
        .copied()
        .map(|n| {
            let h = (b - a) / n as f64;
            let q = piecewise_linear_values(a, b, n, &z);
            let err = z
                .iter()
                .copied()
                .zip(q)
                .map(|(x, qx)| (qx - f(x)).abs())
                .fold(0.0, f64::max);

            (n, h, err)
        })
        .scan(None, |prev_err: &mut Option<f64>, (n, h, err)| {
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow { n, h, err, rate })
        })
        .collect()
}

pub fn smooth_summary() -> Vec<SummaryRow> {
    summarize_case(1.0, 2.0)
}

pub fn singular_summary() -> Vec<SummaryRow> {
    summarize_case(0.0, 1.0)
}

fn write_plot_data() {
    let out_dir = String::from("data/ch2_4");
    let x = fine_grid(PLOT_A, PLOT_B);
    let exact: Vec<f64> = x.iter().copied().map(f).collect();
    let approx = piecewise_linear_values(PLOT_A, PLOT_B, PLOT_N, &x);
    let nodes_x: Vec<f64> = (0..=PLOT_N)
        .map(|i| PLOT_A + (PLOT_B - PLOT_A) * i as f64 / PLOT_N as f64)
        .collect();
    let nodes_y: Vec<f64> = nodes_x.iter().copied().map(f).collect();

    util::write_data(&x, out_dir.clone(), String::from("plot__x"));
    util::write_data(&exact, out_dir.clone(), String::from("plot__exact"));
    util::write_data(&approx, out_dir.clone(), String::from("plot__approx"));
    util::write_data(&nodes_x, out_dir.clone(), String::from("plot__nodes_x"));
    util::write_data(&nodes_y, out_dir, String::from("plot__nodes_y"));
}

fn write_summary_data(name: &str, rows: &[SummaryRow]) {
    let out_dir = String::from("data/ch2_4");
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
    let smooth = smooth_summary();
    let singular = singular_summary();

    write_plot_data();
    write_summary_data("smooth", &smooth);
    write_summary_data("singular", &singular);

    util::plot("ch2_4")?;
    util::run_python_script("scripts/ch2_4/make_tables.py")?;
    util::copy_file(
        "plots/ch2_4/approximation.png",
        "reports/ch2_4/figures/approximation.png",
    )?;
    util::copy_file("plots/ch2_4/error.png", "reports/ch2_4/figures/error.png")?;
    util::build_report("reports/ch2_4", "2.4.pdf")
}

#[cfg(test)]
mod tests {
    use super::{
        PLOT_A, PLOT_B, PLOT_N, f, piecewise_linear_values, singular_summary, smooth_summary,
    };

    #[test]
    fn interpolation_matches_mesh_values() {
        let nodes: Vec<f64> = (0..=PLOT_N)
            .map(|i| PLOT_A + (PLOT_B - PLOT_A) * i as f64 / PLOT_N as f64)
            .collect();
        let q = piecewise_linear_values(PLOT_A, PLOT_B, PLOT_N, &nodes);

        for (x, qx) in nodes.iter().copied().zip(q) {
            assert!((qx - f(x)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn smooth_case_converges_second_order() {
        let rows = smooth_summary();
        let last = rows.last().expect("missing smooth rows");

        assert!(rows[1].err < rows[0].err);
        assert!((last.rate.expect("missing smooth rate") - 2.0).abs() < 0.12);
    }

    #[test]
    fn singular_case_converges_like_h_one_third() {
        let rows = singular_summary();
        let last = rows.last().expect("missing singular rows");
        let rate = last.rate.expect("missing singular rate");

        assert!(rows[1].err < rows[0].err);
        assert!(rate > 0.25 && rate < 0.45);
    }
}
