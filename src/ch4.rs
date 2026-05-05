use crate::util;
use std::io;

const FINE_STEPS: usize = 1000;
const EXP1_N_VALUES: [usize; 6] = [2, 4, 8, 16, 32, 64];
const EXP2_N_VALUES: [usize; 4] = [6, 12, 24, 48];
const EXP3_N_VALUES: [usize; 4] = [6, 12, 24, 48];
const GRAD_N_VALUES: [usize; 6] = [5, 10, 20, 40, 80, 160];

type ScalarFn = fn(f64) -> f64;
type DerivativeFn = fn(f64) -> f64;

#[derive(Debug, Clone, Copy)]
struct SummaryRow {
    n: usize,
    h: f64,
    err: f64,
    rate: Option<f64>,
}

#[derive(Debug, Clone)]
struct CubicSplineExpansion {
    a: f64,
    b: f64,
    h: f64,
    n: usize,
    coeffs: Vec<f64>,
    c_minus1: f64,
    c_nplus1: f64,
}

impl CubicSplineExpansion {
    fn coeff_at(&self, i: isize) -> f64 {
        match i {
            -1 => self.c_minus1,
            _ if i == self.n as isize + 1 => self.c_nplus1,
            _ if (0..=self.n as isize).contains(&i) => self.coeffs[i as usize],
            _ => 0.0,
        }
    }

    fn eval(&self, x: f64) -> f64 {
        let cell = if x >= self.b {
            self.n.saturating_sub(1) as isize
        } else if x <= self.a {
            0
        } else {
            ((x - self.a) / self.h).floor() as isize
        };

        let mut sum = 0.0;
        for i in (cell - 1).max(-1)..=(cell + 2).min(self.n as isize + 1) {
            let xi = self.a + i as f64 * self.h;
            sum += self.coeff_at(i) * parent_bspline((x - xi) / self.h);
        }

        sum
    }

    fn values(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().copied().map(|x| self.eval(x)).collect()
    }
}

fn smooth_fn(x: f64) -> f64 {
    x.sin()
}

fn smooth_df(x: f64) -> f64 {
    x.cos()
}

fn nonsmooth_fn(x: f64) -> f64 {
    (x * x).cbrt()
}

fn nonsmooth_df(x: f64) -> f64 {
    2.0 / (3.0 * x.cbrt())
}

fn runge_fn(x: f64) -> f64 {
    1.0 / (1.0 + 2.0 * x * x)
}

fn runge_df(x: f64) -> f64 {
    -4.0 * x / (1.0 + 2.0 * x * x).powi(2)
}

fn grad_rhs(x: f64) -> f64 {
    4.0 * (-x).exp() - 4.0 * x * (-x).exp()
}

fn grad_exact(x: f64) -> f64 {
    x * (1.0 - x) * (-x).exp()
}

fn fine_grid(a: f64, b: f64) -> Vec<f64> {
    (0..=FINE_STEPS)
        .map(|i| a + (b - a) * i as f64 / FINE_STEPS as f64)
        .collect()
}

fn uniform_nodes(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect()
}

fn chebyshev_nodes(a: f64, b: f64, n: usize) -> Vec<f64> {
    let mut xs: Vec<f64> = (0..=n)
        .map(|i| {
            0.5 * (a + b)
                + 0.5
                    * (b - a)
                    * (((2 * i + 1) as f64 * std::f64::consts::PI) / (2 * n + 2) as f64).cos()
        })
        .collect();
    xs.sort_by(|x, y| x.total_cmp(y));
    xs
}

fn observed_rate(prev_err: f64, err: f64) -> Option<f64> {
    if prev_err == 0.0 || err == 0.0 {
        None
    } else {
        Some((prev_err / err).ln() / 2.0_f64.ln())
    }
}

fn summarize_errors(ns: &[usize], a: f64, b: f64, errors: &[f64]) -> Vec<SummaryRow> {
    ns.iter()
        .copied()
        .zip(errors.iter().copied())
        .scan(None, |prev_err: &mut Option<f64>, (n, err)| {
            let h = (b - a) / n as f64;
            let rate = prev_err.and_then(|prev| observed_rate(prev, err));
            *prev_err = Some(err);

            Some(SummaryRow { n, h, err, rate })
        })
        .collect()
}

fn max_abs_error(values: &[f64], exact: &[f64]) -> f64 {
    values
        .iter()
        .zip(exact)
        .map(|(value, exact_value)| (value - exact_value).abs())
        .fold(0.0, f64::max)
}

fn pointwise_error(values: &[f64], exact: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(exact)
        .map(|(value, exact_value)| value - exact_value)
        .collect()
}

fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    assert_eq!(xs.len(), ys.len(), "divided difference length mismatch");
    let mut coeffs = ys.to_vec();

    for j in 1..xs.len() {
        for i in (j..xs.len()).rev() {
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xs[i] - xs[i - j]);
        }
    }

    coeffs
}

fn evaluate_newton(xs: &[f64], coeffs: &[f64], x: f64) -> f64 {
    let mut value = *coeffs.last().expect("Newton polynomial needs coefficients");

    for i in (0..coeffs.len() - 1).rev() {
        value = value * (x - xs[i]) + coeffs[i];
    }

    value
}

fn lagrange_values(xs: &[f64], f: ScalarFn, domain: &[f64]) -> Vec<f64> {
    let ys: Vec<f64> = xs.iter().copied().map(f).collect();
    let coeffs = divided_differences(xs, &ys);

    domain
        .iter()
        .copied()
        .map(|x| evaluate_newton(xs, &coeffs, x))
        .collect()
}

fn hermite_values(xs: &[f64], f: ScalarFn, df: DerivativeFn, domain: &[f64]) -> Vec<f64> {
    let m = 2 * xs.len();
    let mut xx = vec![0.0; m];
    let mut table = vec![vec![0.0; m]; m];

    for (i, &x) in xs.iter().enumerate() {
        let j = 2 * i;
        xx[j] = x;
        xx[j + 1] = x;
        table[j][0] = f(x);
        table[j + 1][0] = f(x);
        table[j + 1][1] = df(x);

        if i > 0 {
            table[j][1] = (table[j][0] - table[j - 1][0]) / (xx[j] - xx[j - 1]);
        }
    }

    for i in 2..m {
        for j in 2..=i {
            table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (xx[i] - xx[i - j]);
        }
    }

    let coeffs: Vec<f64> = (0..m).map(|i| table[i][i]).collect();
    domain
        .iter()
        .copied()
        .map(|x| evaluate_newton(&xx, &coeffs, x))
        .collect()
}

fn piecewise_lagrange_values(
    a: f64,
    b: f64,
    n: usize,
    degree: usize,
    f: ScalarFn,
    domain: &[f64],
) -> Vec<f64> {
    assert!(degree > 0, "piecewise degree must be positive");
    assert_eq!(n % degree, 0, "n must be divisible by the piecewise degree");

    let nodes = uniform_nodes(a, b, n);
    let pieces = n / degree;
    let h = (b - a) / n as f64;
    let mut local_nodes = Vec::with_capacity(pieces);
    let mut local_coeffs = Vec::with_capacity(pieces);

    for piece in 0..pieces {
        let start = piece * degree;
        let xs = nodes[start..=start + degree].to_vec();
        let ys: Vec<f64> = xs.iter().copied().map(f).collect();
        local_coeffs.push(divided_differences(&xs, &ys));
        local_nodes.push(xs);
    }

    domain
        .iter()
        .copied()
        .map(|x| {
            let piece = if x >= b {
                pieces - 1
            } else {
                (((x - a) / (degree as f64 * h)).floor() as usize).min(pieces - 1)
            };

            evaluate_newton(&local_nodes[piece], &local_coeffs[piece], x)
        })
        .collect()
}

fn parent_bspline(x: f64) -> f64 {
    if x <= -2.0 || x >= 2.0 {
        0.0
    } else if x <= -1.0 {
        (x + 2.0).powi(3)
    } else if x <= 0.0 {
        let t = x + 1.0;
        1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t.powi(3)
    } else if x <= 1.0 {
        let t = 1.0 - x;
        1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t.powi(3)
    } else {
        (2.0 - x).powi(3)
    }
}

fn solve_tridiagonal(
    lower: &[f64],
    mut diag: Vec<f64>,
    upper: &[f64],
    mut rhs: Vec<f64>,
) -> Vec<f64> {
    let n = diag.len();
    assert!(n > 0, "tridiagonal system must be non-empty");
    assert_eq!(lower.len(), n, "lower diagonal length mismatch");
    assert_eq!(upper.len(), n, "upper diagonal length mismatch");
    assert_eq!(rhs.len(), n, "rhs length mismatch");

    for i in 1..n {
        let multiplier = lower[i] / diag[i - 1];
        diag[i] -= multiplier * upper[i - 1];
        rhs[i] -= multiplier * rhs[i - 1];
    }

    let mut solution = vec![0.0; n];
    solution[n - 1] = rhs[n - 1] / diag[n - 1];

    for i in (0..n - 1).rev() {
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diag[i];
    }

    solution
}

fn complete_bspline_interpolant(
    a: f64,
    b: f64,
    n: usize,
    f: ScalarFn,
    df: DerivativeFn,
) -> CubicSplineExpansion {
    assert!(n >= 2, "spline interpolation requires n >= 2");
    let h = (b - a) / n as f64;
    let m = n + 1;
    let nodes = uniform_nodes(a, b, n);
    let mut lower = vec![0.0; m];
    let diag = vec![4.0; m];
    let mut upper = vec![0.0; m];
    let mut rhs = vec![0.0; m];

    upper[0] = 2.0;
    lower[m - 1] = 2.0;
    for i in 1..m - 1 {
        lower[i] = 1.0;
        upper[i] = 1.0;
    }

    rhs[0] = f(nodes[0]) + h * df(nodes[0]) / 3.0;
    rhs[m - 1] = f(nodes[n]) - h * df(nodes[n]) / 3.0;
    for i in 1..m - 1 {
        rhs[i] = f(nodes[i]);
    }

    let coeffs = solve_tridiagonal(&lower, diag, &upper, rhs);
    let c_minus1 = coeffs[1] - h * df(nodes[0]) / 3.0;
    let c_nplus1 = coeffs[n - 1] + h * df(nodes[n]) / 3.0;

    CubicSplineExpansion {
        a,
        b,
        h,
        n,
        coeffs,
        c_minus1,
        c_nplus1,
    }
}

fn collocation_interpolant(n: usize, rhs: ScalarFn) -> CubicSplineExpansion {
    assert!(n >= 2, "collocation requires n >= 2");
    let a = 0.0;
    let b = 1.0;
    let h = 1.0 / n as f64;
    let m = n + 1;
    let nodes = uniform_nodes(a, b, n);
    let alpha = 1.0 - 6.0 / (h * h);
    let beta = 4.0 + 12.0 / (h * h);

    let mut lower = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut load = vec![0.0; m];

    diag[0] = 36.0 / (h * h);
    diag[m - 1] = 36.0 / (h * h);
    load[0] = rhs(nodes[0]);
    load[m - 1] = rhs(nodes[n]);

    for i in 1..m - 1 {
        lower[i] = alpha;
        diag[i] = beta;
        upper[i] = alpha;
        load[i] = rhs(nodes[i]);
    }

    let coeffs = solve_tridiagonal(&lower, diag, &upper, load);
    let c_minus1 = -4.0 * coeffs[0] - coeffs[1];
    let c_nplus1 = -coeffs[n - 1] - 4.0 * coeffs[n];

    CubicSplineExpansion {
        a,
        b,
        h,
        n,
        coeffs,
        c_minus1,
        c_nplus1,
    }
}

fn write_data(data: &[f64], name: &str) {
    util::write_data(data, String::from("data/ch4"), String::from(name));
}

fn write_summary(prefix: &str, rows: &[SummaryRow]) {
    let n: Vec<f64> = rows.iter().map(|row| row.n as f64).collect();
    let h: Vec<f64> = rows.iter().map(|row| row.h).collect();
    let err: Vec<f64> = rows.iter().map(|row| row.err).collect();
    let rate: Vec<f64> = rows
        .iter()
        .map(|row| row.rate.unwrap_or(f64::NAN))
        .collect();

    write_data(&n, &format!("{prefix}__n"));
    write_data(&h, &format!("{prefix}__h"));
    write_data(&err, &format!("{prefix}__err"));
    write_data(&rate, &format!("{prefix}__rate"));
}

fn experiment1() {
    let a = -3.0;
    let b = 3.0;
    let domain = fine_grid(a, b);
    let exact: Vec<f64> = domain.iter().copied().map(smooth_fn).collect();

    let mut lagrange_uniform_err = Vec::with_capacity(EXP1_N_VALUES.len());
    let mut lagrange_cheb_err = Vec::with_capacity(EXP1_N_VALUES.len());
    let mut hermite_err = Vec::with_capacity(EXP1_N_VALUES.len());
    let mut piecewise_err = Vec::with_capacity(EXP1_N_VALUES.len());
    let mut spline_err = Vec::with_capacity(EXP1_N_VALUES.len());

    for &n in &EXP1_N_VALUES {
        let uniform = uniform_nodes(a, b, n);
        let cheb = chebyshev_nodes(a, b, n);
        let lagrange_uniform = lagrange_values(&uniform, smooth_fn, &domain);
        let lagrange_cheb = lagrange_values(&cheb, smooth_fn, &domain);
        let hermite = hermite_values(&uniform, smooth_fn, smooth_df, &domain);
        let piecewise = piecewise_lagrange_values(a, b, n, 2, smooth_fn, &domain);
        let spline = complete_bspline_interpolant(a, b, n, smooth_fn, smooth_df).values(&domain);

        lagrange_uniform_err.push(max_abs_error(&lagrange_uniform, &exact));
        lagrange_cheb_err.push(max_abs_error(&lagrange_cheb, &exact));
        hermite_err.push(max_abs_error(&hermite, &exact));
        piecewise_err.push(max_abs_error(&piecewise, &exact));
        spline_err.push(max_abs_error(&spline, &exact));
    }

    write_summary(
        "exp1__lagrange_uniform",
        &summarize_errors(&EXP1_N_VALUES, a, b, &lagrange_uniform_err),
    );
    write_summary(
        "exp1__lagrange_chebyshev",
        &summarize_errors(&EXP1_N_VALUES, a, b, &lagrange_cheb_err),
    );
    write_summary(
        "exp1__hermite",
        &summarize_errors(&EXP1_N_VALUES, a, b, &hermite_err),
    );
    write_summary(
        "exp1__piecewise_quadratic",
        &summarize_errors(&EXP1_N_VALUES, a, b, &piecewise_err),
    );
    write_summary(
        "exp1__bspline",
        &summarize_errors(&EXP1_N_VALUES, a, b, &spline_err),
    );
}

fn experiment2() {
    let a = -1.0;
    let b = 1.0;
    let domain = fine_grid(a, b);
    let exact: Vec<f64> = domain.iter().copied().map(nonsmooth_fn).collect();

    let mut lagrange_err = Vec::with_capacity(EXP2_N_VALUES.len());
    let mut piecewise_err = Vec::with_capacity(EXP2_N_VALUES.len());
    let mut spline_err = Vec::with_capacity(EXP2_N_VALUES.len());

    for &n in &EXP2_N_VALUES {
        let uniform = uniform_nodes(a, b, n);
        let lagrange = lagrange_values(&uniform, nonsmooth_fn, &domain);
        let piecewise = piecewise_lagrange_values(a, b, n, 3, nonsmooth_fn, &domain);
        let spline =
            complete_bspline_interpolant(a, b, n, nonsmooth_fn, nonsmooth_df).values(&domain);

        lagrange_err.push(max_abs_error(&lagrange, &exact));
        piecewise_err.push(max_abs_error(&piecewise, &exact));
        spline_err.push(max_abs_error(&spline, &exact));

        if n == 6 {
            write_data(&domain, "exp2_plot__x");
            write_data(&exact, "exp2_plot__exact");
            write_data(&lagrange, "exp2_plot__lagrange");
            write_data(&piecewise, "exp2_plot__piecewise_cubic");
            write_data(&spline, "exp2_plot__bspline");
        }

        if n == 12 {
            write_data(&domain, "exp2_error__x");
            write_data(&pointwise_error(&lagrange, &exact), "exp2_error__lagrange");
            write_data(
                &pointwise_error(&piecewise, &exact),
                "exp2_error__piecewise_cubic",
            );
            write_data(&pointwise_error(&spline, &exact), "exp2_error__bspline");
        }
    }

    write_summary(
        "exp2__lagrange",
        &summarize_errors(&EXP2_N_VALUES, a, b, &lagrange_err),
    );
    write_summary(
        "exp2__piecewise_cubic",
        &summarize_errors(&EXP2_N_VALUES, a, b, &piecewise_err),
    );
    write_summary(
        "exp2__bspline",
        &summarize_errors(&EXP2_N_VALUES, a, b, &spline_err),
    );
}

fn experiment3_case(b: f64, tag: &str) {
    let a = -b;
    let domain = fine_grid(a, b);
    let exact: Vec<f64> = domain.iter().copied().map(runge_fn).collect();

    let mut lagrange_uniform_err = Vec::with_capacity(EXP3_N_VALUES.len());
    let mut lagrange_cheb_err = Vec::with_capacity(EXP3_N_VALUES.len());
    let mut piecewise_err = Vec::with_capacity(EXP3_N_VALUES.len());
    let mut spline_err = Vec::with_capacity(EXP3_N_VALUES.len());

    for &n in &EXP3_N_VALUES {
        let uniform = uniform_nodes(a, b, n);
        let cheb = chebyshev_nodes(a, b, n);
        let lagrange_uniform = lagrange_values(&uniform, runge_fn, &domain);
        let lagrange_cheb = lagrange_values(&cheb, runge_fn, &domain);
        let piecewise = piecewise_lagrange_values(a, b, n, 3, runge_fn, &domain);
        let spline = complete_bspline_interpolant(a, b, n, runge_fn, runge_df).values(&domain);

        lagrange_uniform_err.push(max_abs_error(&lagrange_uniform, &exact));
        lagrange_cheb_err.push(max_abs_error(&lagrange_cheb, &exact));
        piecewise_err.push(max_abs_error(&piecewise, &exact));
        spline_err.push(max_abs_error(&spline, &exact));

        if n == 6 {
            write_data(&domain, &format!("exp3_{tag}_plot__x"));
            write_data(&exact, &format!("exp3_{tag}_plot__exact"));
            write_data(
                &lagrange_uniform,
                &format!("exp3_{tag}_plot__lagrange_uniform"),
            );
            write_data(
                &lagrange_cheb,
                &format!("exp3_{tag}_plot__lagrange_chebyshev"),
            );
            write_data(&piecewise, &format!("exp3_{tag}_plot__piecewise_cubic"));
            write_data(&spline, &format!("exp3_{tag}_plot__bspline"));
        }

        if n == 12 {
            write_data(&domain, &format!("exp3_{tag}_error__x"));
            write_data(
                &pointwise_error(&lagrange_uniform, &exact),
                &format!("exp3_{tag}_error__lagrange_uniform"),
            );
            write_data(
                &pointwise_error(&lagrange_cheb, &exact),
                &format!("exp3_{tag}_error__lagrange_chebyshev"),
            );
            write_data(
                &pointwise_error(&piecewise, &exact),
                &format!("exp3_{tag}_error__piecewise_cubic"),
            );
            write_data(
                &pointwise_error(&spline, &exact),
                &format!("exp3_{tag}_error__bspline"),
            );
        }
    }

    write_summary(
        &format!("exp3_{tag}__lagrange_uniform"),
        &summarize_errors(&EXP3_N_VALUES, a, b, &lagrange_uniform_err),
    );
    write_summary(
        &format!("exp3_{tag}__lagrange_chebyshev"),
        &summarize_errors(&EXP3_N_VALUES, a, b, &lagrange_cheb_err),
    );
    write_summary(
        &format!("exp3_{tag}__piecewise_cubic"),
        &summarize_errors(&EXP3_N_VALUES, a, b, &piecewise_err),
    );
    write_summary(
        &format!("exp3_{tag}__bspline"),
        &summarize_errors(&EXP3_N_VALUES, a, b, &spline_err),
    );
}

fn graduate_section() {
    let domain = fine_grid(0.0, 1.0);
    let exact: Vec<f64> = domain.iter().copied().map(grad_exact).collect();
    let mut errors = Vec::with_capacity(GRAD_N_VALUES.len());

    for &n in &GRAD_N_VALUES {
        let spline = collocation_interpolant(n, grad_rhs);
        let approx = spline.values(&domain);
        let err = pointwise_error(&approx, &exact);
        errors.push(max_abs_error(&approx, &exact));

        if n == 5 {
            write_data(&domain, "grad_plot__x");
            write_data(&exact, "grad_plot__exact");
            write_data(&approx, "grad_plot__approx");
        }

        if n == 20 {
            write_data(&domain, "grad_error__x");
            write_data(&err, "grad_error__err");
        }
    }

    write_summary("grad", &summarize_errors(&GRAD_N_VALUES, 0.0, 1.0, &errors));
}

pub fn generate() -> io::Result<()> {
    experiment1();
    experiment2();
    experiment3_case(1.0, "b1");
    experiment3_case(4.0, "b4");
    graduate_section();

    util::plot("ch4")?;
    util::run_python_script("scripts/ch4/make_tables.py")?;

    for figure in [
        "exp1_convergence.png",
        "exp2_approximation.png",
        "exp2_error.png",
        "exp3_b1_approximation.png",
        "exp3_b1_error.png",
        "exp3_b4_approximation.png",
        "exp3_b4_error.png",
        "grad_approximation.png",
        "grad_error.png",
    ] {
        util::copy_file(
            &format!("plots/ch4/{figure}"),
            &format!("reports/ch4/figures/{figure}"),
        )?;
    }

    util::build_report("reports/ch4", "4.pdf")
}

#[cfg(test)]
mod tests {
    use super::{
        complete_bspline_interpolant, grad_exact, grad_rhs, hermite_values, lagrange_values,
        piecewise_lagrange_values, uniform_nodes,
    };

    fn quad(x: f64) -> f64 {
        x * x - 2.0 * x + 3.0
    }

    fn cubic(x: f64) -> f64 {
        x * x * x - 2.0 * x + 1.0
    }

    fn cubic_df(x: f64) -> f64 {
        3.0 * x * x - 2.0
    }

    #[test]
    fn lagrange_recovers_quadratic_exactly() {
        let nodes = uniform_nodes(-1.0, 1.0, 2);
        let domain = [-0.75, -0.2, 0.0, 0.3, 0.9];
        let values = lagrange_values(&nodes, quad, &domain);

        for (x, value) in domain.into_iter().zip(values) {
            assert!((value - quad(x)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn hermite_recovers_cubic_exactly() {
        let nodes = uniform_nodes(-1.0, 1.0, 1);
        let domain = [-1.0, -0.25, 0.0, 0.5, 1.0];
        let values = hermite_values(&nodes, cubic, cubic_df, &domain);

        for (x, value) in domain.into_iter().zip(values) {
            assert!((value - cubic(x)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn piecewise_quadratic_recovers_quadratic_exactly() {
        let domain = [-1.0, -0.4, 0.2, 0.7, 1.0];
        let values = piecewise_lagrange_values(-1.0, 1.0, 4, 2, quad, &domain);

        for (x, value) in domain.into_iter().zip(values) {
            assert!((value - quad(x)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn spline_reproduces_linear_data() {
        fn linear(x: f64) -> f64 {
            2.0 * x - 1.0
        }

        fn linear_df(_: f64) -> f64 {
            2.0
        }

        let spline = complete_bspline_interpolant(-1.0, 1.0, 4, linear, linear_df);
        let domain = [-1.0, -0.5, 0.0, 0.25, 1.0];

        for x in domain {
            assert!((spline.eval(x) - linear(x)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn collocation_solution_matches_boundary_conditions() {
        let spline = super::collocation_interpolant(10, grad_rhs);
        assert!(spline.eval(0.0).abs() < 1.0e-12);
        assert!(spline.eval(1.0).abs() < 1.0e-12);
        assert!(spline.eval(0.5).is_finite());
        assert!(grad_exact(0.5).is_finite());
    }
}
