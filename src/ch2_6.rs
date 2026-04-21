use crate::util;
use std::io;

const N_VALUES: [usize; 4] = [3, 10, 25, 100];

#[derive(Debug, Clone)]
pub struct SummaryRow {
    pub n: usize,
    pub en: f64,
}

#[derive(Debug, Clone)]
struct CaseData {
    n: usize,
    solution: Vec<f64>,
    residual: Vec<f64>,
    en: f64,
}

fn build_problem(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(n > 0, "n must be positive");

    let mut lower = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let rhs = vec![1.0; n];

    for i in 0..n {
        if i > 0 {
            lower[i] = 1.0;
        }
        diag[i] = i as f64 + 2.0;
        if i + 1 < n {
            upper[i] = 1.0;
        }
    }

    (lower, diag, upper, rhs)
}

fn solve_tridiagonal(
    lower: &[f64],
    mut diag: Vec<f64>,
    upper: &[f64],
    mut rhs: Vec<f64>,
) -> Vec<f64> {
    let n = diag.len();
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

fn tridiagonal_matvec(lower: &[f64], diag: &[f64], upper: &[f64], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert_eq!(lower.len(), n, "lower diagonal length mismatch");
    assert_eq!(diag.len(), n, "diagonal length mismatch");
    assert_eq!(upper.len(), n, "upper diagonal length mismatch");

    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut value = diag[i] * y[i];
        if i > 0 {
            value += lower[i] * y[i - 1];
        }
        if i + 1 < n {
            value += upper[i] * y[i + 1];
        }
        out[i] = value;
    }

    out
}

fn vector_subtract(y: &[f64], z: &[f64]) -> Vec<f64> {
    assert_eq!(y.len(), z.len(), "vector length mismatch");
    y.iter().zip(z).map(|(yi, zi)| yi - zi).collect()
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

fn solve_case(n: usize) -> CaseData {
    let (lower, diag, upper, rhs) = build_problem(n);
    let solution = solve_tridiagonal(&lower, diag, &upper, rhs);

    let (lower_r, diag_r, upper_r, rhs_r) = build_problem(n);
    let tx = tridiagonal_matvec(&lower_r, &diag_r, &upper_r, &solution);
    let residual = vector_subtract(&rhs_r, &tx);
    let en = max_abs(&residual);

    CaseData {
        n,
        solution,
        residual,
        en,
    }
}

fn generate_cases() -> Vec<CaseData> {
    N_VALUES.into_iter().map(solve_case).collect()
}

fn write_summary_data(rows: &[SummaryRow]) {
    let out_dir = String::from("data/ch2_6");
    let n: Vec<f64> = rows.iter().map(|row| row.n as f64).collect();
    let en: Vec<f64> = rows.iter().map(|row| row.en).collect();

    util::write_data(&n, out_dir.clone(), String::from("summary__n"));
    util::write_data(&en, out_dir, String::from("summary__en"));
}

fn write_profile_data(cases: &[CaseData]) {
    let out_dir = String::from("data/ch2_6");

    for case in cases {
        let indices: Vec<f64> = (1..=case.n).map(|i| i as f64).collect();
        let abs_residual: Vec<f64> = case.residual.iter().copied().map(f64::abs).collect();

        util::write_data(
            &indices,
            out_dir.clone(),
            format!("residual__n{}__i", case.n),
        );
        util::write_data(
            &abs_residual,
            out_dir.clone(),
            format!("residual__n{}__abs", case.n),
        );
        util::write_data(
            &case.solution,
            out_dir.clone(),
            format!("solution__n{}__x", case.n),
        );
        util::write_data(
            &indices,
            out_dir.clone(),
            format!("solution__n{}__i", case.n),
        );
    }
}

pub fn generate() -> io::Result<()> {
    let cases = generate_cases();
    let summary: Vec<SummaryRow> = cases
        .iter()
        .map(|case| SummaryRow {
            n: case.n,
            en: case.en,
        })
        .collect();

    for row in &summary {
        println!("n = {:>3}, e_n = {:.10e}", row.n, row.en);
    }

    write_summary_data(&summary);
    write_profile_data(&cases);

    util::plot("ch2_6")?;
    util::run_python_script("scripts/ch2_6/make_tables.py")?;
    util::copy_file(
        "plots/ch2_6/residual_profiles.png",
        "reports/ch2_6/figures/residual_profiles.png",
    )?;
    util::copy_file(
        "plots/ch2_6/max_residual.png",
        "reports/ch2_6/figures/max_residual.png",
    )?;
    util::build_report("reports/ch2_6", "2.6.pdf")
}

#[cfg(test)]
mod tests {
    use super::{
        N_VALUES, build_problem, max_abs, solve_case, solve_tridiagonal, tridiagonal_matvec,
        vector_subtract,
    };

    #[test]
    fn solves_n3_reference_system() {
        let (lower, diag, upper, rhs) = build_problem(3);
        let x = solve_tridiagonal(&lower, diag, &upper, rhs);
        let expected = [4.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0];

        for (xi, expected_i) in x.iter().copied().zip(expected) {
            assert!((xi - expected_i).abs() < 1.0e-12);
        }
    }

    #[test]
    fn matvec_and_subtraction_produce_small_residuals() {
        for n in N_VALUES {
            let case = solve_case(n);
            assert!(
                max_abs(&case.residual) < 1.0e-12,
                "residual too large for n={n}"
            );
        }
    }

    #[test]
    fn tridiagonal_matvec_matches_problem_definition() {
        let (lower, diag, upper, _) = build_problem(4);
        let y = [1.0, -2.0, 0.5, 3.0];
        let w = tridiagonal_matvec(&lower, &diag, &upper, &y);
        let expected = [
            2.0 * y[0] + y[1],
            y[0] + 3.0 * y[1] + y[2],
            y[1] + 4.0 * y[2] + y[3],
            y[2] + 5.0 * y[3],
        ];

        for (wi, expected_i) in w.iter().copied().zip(expected) {
            assert!((wi - expected_i).abs() < 1.0e-12);
        }

        let residual = vector_subtract(&expected, &w);
        assert!(max_abs(&residual) < 1.0e-12);
    }
}
