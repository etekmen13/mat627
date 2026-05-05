use crate::util;
use std::fs;
use std::io;

const DEFAULT_TOL: f64 = 1.0e-6;
const EXP3_TOL: f64 = 1.0e-4;
const MAX_ITER: usize = 1000;
const PLOT_STEPS: usize = 1200;
const RATE_TOL: f64 = 1.0e-14;
const GRAD_TOL: f64 = 1.0e-16;
const POLY9_COEFFS: [f64; 10] = [
    1.0, -18.0, 144.0, -672.0, 2016.0, -4032.0, 5376.0, -4608.0, 2304.0, -512.0,
];

type ScalarFn = fn(f64) -> f64;

#[derive(Clone, Copy)]
struct FunctionSuite {
    f: ScalarFn,
    df: ScalarFn,
    ddf: ScalarFn,
}

impl FunctionSuite {
    fn f(self, x: f64) -> f64 {
        (self.f)(x)
    }

    fn df(self, x: f64) -> f64 {
        (self.df)(x)
    }

    fn ddf(self, x: f64) -> f64 {
        (self.ddf)(x)
    }
}

#[derive(Debug, Clone)]
struct BracketRow {
    n: usize,
    a: f64,
    b: f64,
    x: f64,
    half_width: f64,
    error: Option<f64>,
}

#[derive(Debug, Clone)]
struct IterateRow {
    n: usize,
    step: Option<f64>,
    fx_abs: f64,
    x: f64,
    error: Option<f64>,
    rate: Option<f64>,
}

#[derive(Debug, Clone)]
struct BracketResult {
    rows: Vec<BracketRow>,
    status: String,
}

#[derive(Debug, Clone)]
struct IterateResult {
    rows: Vec<IterateRow>,
    status: String,
}

#[derive(Debug, Clone)]
struct SafeBracketRow {
    n: usize,
    a: f64,
    b: f64,
    x: f64,
    half_width: f64,
    error: f64,
    value: f64,
    bound: f64,
    sign_code: f64,
}

#[derive(Debug, Clone)]
struct SafeBracketResult {
    rows: Vec<SafeBracketRow>,
    status: String,
}

#[derive(Debug, Clone, Copy)]
struct GradScanRow {
    a: f64,
    b: f64,
    root: f64,
    iterations: usize,
    residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignClass {
    Negative,
    Positive,
    Uncertain,
}

fn experiment1_suite() -> FunctionSuite {
    FunctionSuite {
        f: exp1_f,
        df: exp1_df,
        ddf: exp1_ddf,
    }
}

fn experiment2_suite() -> FunctionSuite {
    FunctionSuite {
        f: exp2_f,
        df: exp2_df,
        ddf: exp2_ddf,
    }
}

fn experiment3_suite() -> FunctionSuite {
    FunctionSuite {
        f: exp3_f,
        df: exp3_df,
        ddf: exp3_ddf,
    }
}

fn exp1_f(x: f64) -> f64 {
    horner(&[0.04, -0.44, 0.61, 2.70, -5.25], x)
}

fn exp1_df(x: f64) -> f64 {
    horner(&[0.16, -1.32, 1.22, 2.70], x)
}

fn exp1_ddf(x: f64) -> f64 {
    horner(&[0.48, -2.64, 1.22], x)
}

fn exp2_f(x: f64) -> f64 {
    2.0 - x.exp()
}

fn exp2_df(x: f64) -> f64 {
    -x.exp()
}

fn exp2_ddf(x: f64) -> f64 {
    -x.exp()
}

fn exp3_f(x: f64) -> f64 {
    10.0 * x * (-(x * x)).exp()
}

fn exp3_df(x: f64) -> f64 {
    10.0 * (1.0 - 2.0 * x * x) * (-(x * x)).exp()
}

fn exp3_ddf(x: f64) -> f64 {
    20.0 * x * (2.0 * x * x - 3.0) * (-(x * x)).exp()
}

fn poly9(x: f64) -> f64 {
    horner(&POLY9_COEFFS, x)
}

fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs
        .iter()
        .copied()
        .reduce(|acc, coeff| acc * x + coeff)
        .expect("coefficients must be non-empty")
}

fn horner_with_bound(coeffs: &[f64], x: f64) -> (f64, f64) {
    let degree = coeffs.len() - 1;
    let mut value = coeffs[0];
    let mut d = coeffs[0].abs();

    for &coeff in &coeffs[1..] {
        value = value * x + coeff;
        d = x.abs() * d + coeff.abs();
    }

    let bound = 2.0 * degree as f64 * f64::EPSILON * d;
    (value, bound)
}

fn classify_sign(value: f64, bound: f64) -> SignClass {
    if value - bound > 0.0 {
        SignClass::Positive
    } else if value + bound < 0.0 {
        SignClass::Negative
    } else {
        SignClass::Uncertain
    }
}

fn sign_code(sign: SignClass) -> f64 {
    match sign {
        SignClass::Negative => -1.0,
        SignClass::Positive => 1.0,
        SignClass::Uncertain => 0.0,
    }
}

fn observed_order(q_nm2: f64, q_nm1: f64, q_n: f64) -> Option<f64> {
    let a = q_nm2.abs();
    let b = q_nm1.abs();
    let c = q_n.abs();
    if a < RATE_TOL || b < RATE_TOL || c < RATE_TOL {
        return None;
    }

    let denominator = (b / a).ln();
    if denominator.abs() < RATE_TOL {
        return None;
    }

    let order = (c / b).ln() / denominator;
    order.is_finite().then_some(order)
}

fn iterate_rate(rows: &[IterateRow], x_new: f64, alpha: Option<f64>) -> Option<f64> {
    match alpha {
        Some(alpha) if rows.len() >= 2 => observed_order(
            rows[rows.len() - 2].x - alpha,
            rows[rows.len() - 1].x - alpha,
            x_new - alpha,
        ),
        None if rows.len() >= 3 => observed_order(
            rows[rows.len() - 2].x - rows[rows.len() - 3].x,
            rows[rows.len() - 1].x - rows[rows.len() - 2].x,
            x_new - rows[rows.len() - 1].x,
        ),
        _ => None,
    }
}

fn iterate_row(rows: &[IterateRow], x: f64, fx: f64, alpha: Option<f64>) -> IterateRow {
    IterateRow {
        n: rows.len(),
        step: rows.last().map(|prev| (x - prev.x).abs()),
        fx_abs: fx.abs(),
        x,
        error: alpha.map(|root| (x - root).abs()),
        rate: iterate_rate(rows, x, alpha),
    }
}

fn bracket_row(n: usize, a: f64, b: f64, x: f64, alpha: Option<f64>) -> BracketRow {
    BracketRow {
        n,
        a,
        b,
        x,
        half_width: 0.5 * (b - a).abs(),
        error: alpha.map(|root| (x - root).abs()),
    }
}

fn format_status(base: &str, warnings: &[String]) -> String {
    if warnings.is_empty() {
        String::from(base)
    } else {
        format!("{base}; {}", warnings.join("; "))
    }
}

fn default_stop(step: f64, fx_abs: f64, tol: f64) -> bool {
    step + fx_abs <= tol / 5.0
}

fn bisection(f: ScalarFn, mut a: f64, mut b: f64, tol: f64, alpha: Option<f64>) -> BracketResult {
    let mut rows = Vec::new();
    let mut fa = f(a);
    let fb = f(b);

    if fa == 0.0 {
        rows.push(bracket_row(0, a, a, a, alpha));
        return BracketResult {
            rows,
            status: String::from("exact left endpoint"),
        };
    }

    if fb == 0.0 {
        rows.push(bracket_row(0, b, b, b, alpha));
        return BracketResult {
            rows,
            status: String::from("exact right endpoint"),
        };
    }

    if fa * fb > 0.0 {
        return BracketResult {
            rows,
            status: String::from("error: f(a)f(b) > 0"),
        };
    }

    let mut x = 0.5 * (a + b);
    rows.push(bracket_row(0, a, b, x, alpha));

    while rows.len() - 1 < MAX_ITER && 0.5 * (b - a).abs() > tol {
        let fx = f(x);
        if fx == 0.0 {
            return BracketResult {
                rows,
                status: String::from("exact midpoint root"),
            };
        }

        if fa * fx <= 0.0 {
            b = x;
        } else {
            a = x;
            fa = fx;
        }

        x = 0.5 * (a + b);
        rows.push(bracket_row(rows.len(), a, b, x, alpha));
    }

    let status = if 0.5 * (b - a).abs() <= tol {
        String::from("converged")
    } else {
        String::from("iteration cap reached")
    };

    BracketResult { rows, status }
}

fn newton(
    funcs: FunctionSuite,
    x0: f64,
    tol: f64,
    alpha: Option<f64>,
    fixed_iterations: Option<usize>,
) -> IterateResult {
    let mut rows = vec![IterateRow {
        n: 0,
        step: None,
        fx_abs: funcs.f(x0).abs(),
        x: x0,
        error: alpha.map(|root| (x0 - root).abs()),
        rate: None,
    }];
    let mut warnings = Vec::new();
    let mut warned = false;

    loop {
        let current_n = rows.last().expect("missing row").n;
        if let Some(m) = fixed_iterations {
            if current_n >= m {
                return IterateResult {
                    rows,
                    status: format_status("fixed-iteration run complete", &warnings),
                };
            }
        } else if current_n >= MAX_ITER {
            return IterateResult {
                rows,
                status: format_status("error: performed more than 1000 iterations", &warnings),
            };
        }

        let x = rows.last().expect("missing row").x;
        let fx = funcs.f(x);
        let dfx = funcs.df(x);
        if dfx.abs() < RATE_TOL {
            return IterateResult {
                rows,
                status: format_status("error: derivative underflow in Newton step", &warnings),
            };
        }

        let x_next = x - fx / dfx;
        let row = iterate_row(&rows, x_next, funcs.f(x_next), alpha);
        rows.push(row);

        if !warned && funcs.df(x_next).abs() < tol {
            warnings.push(String::from("warning: |f'(x_n)| < TOL"));
            warned = true;
        }

        let last = rows.last().expect("missing row");
        if fixed_iterations.is_none() && default_stop(last.step.unwrap_or(0.0), last.fx_abs, tol) {
            return IterateResult {
                rows,
                status: format_status("converged", &warnings),
            };
        }
    }
}

fn chord(
    funcs: FunctionSuite,
    x0: f64,
    tol: f64,
    alpha: Option<f64>,
    fixed_iterations: Option<usize>,
) -> IterateResult {
    let mut rows = vec![IterateRow {
        n: 0,
        step: None,
        fx_abs: funcs.f(x0).abs(),
        x: x0,
        error: alpha.map(|root| (x0 - root).abs()),
        rate: None,
    }];
    let mut warnings = Vec::new();
    let mut warned = false;
    let slope = funcs.df(x0);

    if slope.abs() < RATE_TOL {
        return IterateResult {
            rows,
            status: String::from("error: f'(x_0) is too small for chord method"),
        };
    }

    loop {
        let current_n = rows.last().expect("missing row").n;
        if let Some(m) = fixed_iterations {
            if current_n >= m {
                return IterateResult {
                    rows,
                    status: format_status("fixed-iteration run complete", &warnings),
                };
            }
        } else if current_n >= MAX_ITER {
            return IterateResult {
                rows,
                status: format_status("error: performed more than 1000 iterations", &warnings),
            };
        }

        let x = rows.last().expect("missing row").x;
        let x_next = x - funcs.f(x) / slope;
        let row = iterate_row(&rows, x_next, funcs.f(x_next), alpha);
        rows.push(row);

        if !warned && funcs.df(x_next).abs() < tol {
            warnings.push(String::from("warning: |f'(x_n)| < TOL"));
            warned = true;
        }

        let last = rows.last().expect("missing row");
        if fixed_iterations.is_none() && default_stop(last.step.unwrap_or(0.0), last.fx_abs, tol) {
            return IterateResult {
                rows,
                status: format_status("converged", &warnings),
            };
        }
    }
}

fn secant(
    f: ScalarFn,
    x0: f64,
    x1: f64,
    tol: f64,
    alpha: Option<f64>,
    fixed_iterations: Option<usize>,
) -> IterateResult {
    let mut rows = vec![
        IterateRow {
            n: 0,
            step: None,
            fx_abs: f(x0).abs(),
            x: x0,
            error: alpha.map(|root| (x0 - root).abs()),
            rate: None,
        },
        IterateRow {
            n: 1,
            step: Some((x1 - x0).abs()),
            fx_abs: f(x1).abs(),
            x: x1,
            error: alpha.map(|root| (x1 - root).abs()),
            rate: None,
        },
    ];
    let mut warnings = Vec::new();
    let mut warned = false;

    loop {
        let current_n = rows.last().expect("missing row").n;
        if let Some(m) = fixed_iterations {
            if current_n >= m {
                return IterateResult {
                    rows,
                    status: format_status("fixed-iteration run complete", &warnings),
                };
            }
        } else if current_n >= MAX_ITER {
            return IterateResult {
                rows,
                status: format_status("error: performed more than 1000 iterations", &warnings),
            };
        }

        let x_nm1 = rows[rows.len() - 2].x;
        let x_n = rows[rows.len() - 1].x;
        let f_nm1 = f(x_nm1);
        let f_n = f(x_n);
        let denominator = f_n - f_nm1;
        let step = (x_n - x_nm1).abs();

        if !warned && denominator.abs() < tol * step {
            warnings.push(String::from(
                "warning: |f(x_n)-f(x_{n-1})| < TOL |x_n-x_{n-1}|",
            ));
            warned = true;
        }

        if denominator.abs() < RATE_TOL {
            return IterateResult {
                rows,
                status: format_status("error: secant denominator underflow", &warnings),
            };
        }

        let x_next = x_n - f_n * (x_n - x_nm1) / denominator;
        let row = iterate_row(&rows, x_next, f(x_next), alpha);
        rows.push(row);

        let last = rows.last().expect("missing row");
        if fixed_iterations.is_none() && default_stop(last.step.unwrap_or(0.0), last.fx_abs, tol) {
            return IterateResult {
                rows,
                status: format_status("converged", &warnings),
            };
        }
    }
}

fn super_halley(
    funcs: FunctionSuite,
    x0: f64,
    tol: f64,
    alpha: Option<f64>,
    fixed_iterations: Option<usize>,
) -> IterateResult {
    let mut rows = vec![IterateRow {
        n: 0,
        step: None,
        fx_abs: funcs.f(x0).abs(),
        x: x0,
        error: alpha.map(|root| (x0 - root).abs()),
        rate: None,
    }];
    let mut warnings = Vec::new();
    let mut warned = false;

    loop {
        let current_n = rows.last().expect("missing row").n;
        if let Some(m) = fixed_iterations {
            if current_n >= m {
                return IterateResult {
                    rows,
                    status: format_status("fixed-iteration run complete", &warnings),
                };
            }
        } else if current_n >= MAX_ITER {
            return IterateResult {
                rows,
                status: format_status("error: performed more than 1000 iterations", &warnings),
            };
        }

        let x = rows.last().expect("missing row").x;
        let fx = funcs.f(x);
        let dfx = funcs.df(x);
        if dfx.abs() < RATE_TOL {
            return IterateResult {
                rows,
                status: format_status(
                    "error: derivative underflow in super Halley step",
                    &warnings,
                ),
            };
        }

        let w = fx * funcs.ddf(x) / (dfx * dfx);
        let denominator = 1.0 - w;
        if denominator.abs() < RATE_TOL {
            return IterateResult {
                rows,
                status: format_status("error: 1 - w_n = 0 in super Halley step", &warnings),
            };
        }

        let x_next = x - (1.0 + 0.5 * w / denominator) * fx / dfx;
        let row = iterate_row(&rows, x_next, funcs.f(x_next), alpha);
        rows.push(row);

        if !warned && funcs.df(x_next).abs() < tol {
            warnings.push(String::from("warning: |f'(x_n)| < TOL"));
            warned = true;
        }

        let last = rows.last().expect("missing row");
        if fixed_iterations.is_none() && default_stop(last.step.unwrap_or(0.0), last.fx_abs, tol) {
            return IterateResult {
                rows,
                status: format_status("converged", &warnings),
            };
        }
    }
}

fn hybrid(
    funcs: FunctionSuite,
    mut a: f64,
    mut b: f64,
    tol: f64,
    alpha: Option<f64>,
) -> BracketResult {
    let mut rows = Vec::new();
    let mut fa = funcs.f(a);
    let mut fb = funcs.f(b);

    if fa == 0.0 {
        rows.push(bracket_row(0, a, a, a, alpha));
        return BracketResult {
            rows,
            status: String::from("exact left endpoint"),
        };
    }

    if fb == 0.0 {
        rows.push(bracket_row(0, b, b, b, alpha));
        return BracketResult {
            rows,
            status: String::from("exact right endpoint"),
        };
    }

    if fa * fb > 0.0 {
        return BracketResult {
            rows,
            status: String::from("error: f(a)f(b) > 0"),
        };
    }

    rows.push(bracket_row(0, a, b, a, alpha));
    rows.push(bracket_row(1, a, b, b, alpha));
    let mut x = b;

    while rows.len() - 1 < MAX_ITER {
        let midpoint = 0.5 * (a + b);
        let fx = funcs.f(x);
        let dfx = funcs.df(x);
        let mut c = if dfx.abs() >= RATE_TOL {
            x - fx / dfx
        } else {
            midpoint
        };

        if !c.is_finite() || c <= a.min(b) || c >= a.max(b) {
            c = midpoint;
        }

        let fc = funcs.f(c);
        if fa * fc <= 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }

        rows.push(bracket_row(rows.len(), a, b, c, alpha));

        if (c - x).abs() + fc.abs() <= tol / 5.0 || (b - a).abs() <= tol {
            return BracketResult {
                rows,
                status: String::from("converged"),
            };
        }

        x = c;
        if fb == 0.0 {
            let last = rows.last_mut().expect("missing row");
            last.x = b;
            return BracketResult {
                rows,
                status: String::from("exact right endpoint"),
            };
        }
    }

    BracketResult {
        rows,
        status: String::from("iteration cap reached"),
    }
}

fn safe_bisection_with_horner(
    coeffs: &[f64],
    mut a: f64,
    mut b: f64,
    tol: f64,
    alpha: f64,
) -> SafeBracketResult {
    let mut rows = Vec::new();
    let (fa, da) = horner_with_bound(coeffs, a);
    let (fb, db) = horner_with_bound(coeffs, b);
    let mut sign_a = classify_sign(fa, da);
    let sign_b = classify_sign(fb, db);

    if sign_a == SignClass::Uncertain || sign_b == SignClass::Uncertain {
        return SafeBracketResult {
            rows,
            status: String::from("error: uncertain sign at initial endpoint"),
        };
    }

    if sign_a == sign_b {
        return SafeBracketResult {
            rows,
            status: String::from("error: no verified sign change on initial interval"),
        };
    }

    loop {
        let x = 0.5 * (a + b);
        let (value, bound) = horner_with_bound(coeffs, x);
        let sign_x = classify_sign(value, bound);
        rows.push(SafeBracketRow {
            n: rows.len(),
            a,
            b,
            x,
            half_width: 0.5 * (b - a).abs(),
            error: (x - alpha).abs(),
            value,
            bound,
            sign_code: sign_code(sign_x),
        });

        if sign_x == SignClass::Uncertain {
            return SafeBracketResult {
                rows,
                status: String::from("stopped: sign of p(x_n) is unreliable from roundoff"),
            };
        }

        if 0.5 * (b - a).abs() <= tol {
            return SafeBracketResult {
                rows,
                status: String::from("converged before sign uncertainty"),
            };
        }

        if sign_x == sign_a {
            a = x;
            sign_a = sign_x;
        } else {
            b = x;
        }

        if rows.len() >= MAX_ITER {
            return SafeBracketResult {
                rows,
                status: String::from("iteration cap reached"),
            };
        }
    }
}

fn final_bracket_summary(result: &BracketResult, f: ScalarFn) -> (usize, f64, f64) {
    let row = result.rows.last().expect("missing bracket row");
    (row.n, row.x, f(row.x).abs())
}

fn final_iterate_summary(result: &IterateResult) -> (usize, f64, f64, Option<f64>) {
    let row = result.rows.last().expect("missing iterate row");
    (row.n, row.x, row.fx_abs, row.rate)
}

fn linspace(a: f64, b: f64, steps: usize) -> Vec<f64> {
    (0..=steps)
        .map(|i| a + (b - a) * i as f64 / steps as f64)
        .collect()
}

fn write_data(slug: &str, name: &str, values: &[f64]) {
    util::write_data(values, String::from("data/ch3"), format!("{slug}__{name}"));
}

fn write_status(slug: &str, status: &str) -> io::Result<()> {
    fs::create_dir_all("data/ch3")?;
    fs::write(format!("data/ch3/{slug}__status.txt"), status)
}

fn write_bracket_history(slug: &str, result: &BracketResult) -> io::Result<()> {
    let n: Vec<f64> = result.rows.iter().map(|row| row.n as f64).collect();
    let a: Vec<f64> = result.rows.iter().map(|row| row.a).collect();
    let b: Vec<f64> = result.rows.iter().map(|row| row.b).collect();
    let x: Vec<f64> = result.rows.iter().map(|row| row.x).collect();
    let half_width: Vec<f64> = result.rows.iter().map(|row| row.half_width).collect();
    let error: Vec<f64> = result
        .rows
        .iter()
        .map(|row| row.error.unwrap_or(f64::NAN))
        .collect();

    write_data(slug, "n", &n);
    write_data(slug, "a", &a);
    write_data(slug, "b", &b);
    write_data(slug, "x", &x);
    write_data(slug, "half_width", &half_width);
    write_data(slug, "error", &error);
    write_status(slug, &result.status)
}

fn write_iterate_history(slug: &str, result: &IterateResult) -> io::Result<()> {
    let n: Vec<f64> = result.rows.iter().map(|row| row.n as f64).collect();
    let step: Vec<f64> = result
        .rows
        .iter()
        .map(|row| row.step.unwrap_or(f64::NAN))
        .collect();
    let fx_abs: Vec<f64> = result.rows.iter().map(|row| row.fx_abs).collect();
    let x: Vec<f64> = result.rows.iter().map(|row| row.x).collect();
    let error: Vec<f64> = result
        .rows
        .iter()
        .map(|row| row.error.unwrap_or(f64::NAN))
        .collect();
    let rate: Vec<f64> = result
        .rows
        .iter()
        .map(|row| row.rate.unwrap_or(f64::NAN))
        .collect();

    write_data(slug, "n", &n);
    write_data(slug, "step", &step);
    write_data(slug, "fx_abs", &fx_abs);
    write_data(slug, "x", &x);
    write_data(slug, "error", &error);
    write_data(slug, "rate", &rate);
    write_status(slug, &result.status)
}

fn write_safe_history(slug: &str, result: &SafeBracketResult) -> io::Result<()> {
    let n: Vec<f64> = result.rows.iter().map(|row| row.n as f64).collect();
    let a: Vec<f64> = result.rows.iter().map(|row| row.a).collect();
    let b: Vec<f64> = result.rows.iter().map(|row| row.b).collect();
    let x: Vec<f64> = result.rows.iter().map(|row| row.x).collect();
    let half_width: Vec<f64> = result.rows.iter().map(|row| row.half_width).collect();
    let error: Vec<f64> = result.rows.iter().map(|row| row.error).collect();
    let value: Vec<f64> = result.rows.iter().map(|row| row.value).collect();
    let bound: Vec<f64> = result.rows.iter().map(|row| row.bound).collect();
    let sign: Vec<f64> = result.rows.iter().map(|row| row.sign_code).collect();

    write_data(slug, "n", &n);
    write_data(slug, "a", &a);
    write_data(slug, "b", &b);
    write_data(slug, "x", &x);
    write_data(slug, "half_width", &half_width);
    write_data(slug, "error", &error);
    write_data(slug, "value", &value);
    write_data(slug, "bound", &bound);
    write_data(slug, "sign_code", &sign);
    write_status(slug, &result.status)
}

fn write_function_plot(slug: &str, a: f64, b: f64, f: ScalarFn) {
    let x = linspace(a, b, PLOT_STEPS);
    let y: Vec<f64> = x.iter().copied().map(f).collect();
    write_data(slug, "x", &x);
    write_data(slug, "y", &y);
}

fn write_grad_scan(rows: &[GradScanRow]) {
    let a: Vec<f64> = rows.iter().map(|row| row.a).collect();
    let b: Vec<f64> = rows.iter().map(|row| row.b).collect();
    let root: Vec<f64> = rows.iter().map(|row| row.root).collect();
    let iterations: Vec<f64> = rows.iter().map(|row| row.iterations as f64).collect();
    let residual: Vec<f64> = rows.iter().map(|row| row.residual).collect();

    write_data("grad_scan", "a", &a);
    write_data("grad_scan", "b", &b);
    write_data("grad_scan", "root", &root);
    write_data("grad_scan", "iterations", &iterations);
    write_data("grad_scan", "residual", &residual);
}

fn run_experiment1() -> io::Result<()> {
    let suite = experiment1_suite();

    write_function_plot("exp1_function", -3.0, 9.0, suite.f);

    let neg_bisection = bisection(suite.f, -3.0, -2.0, DEFAULT_TOL, None);
    let neg_newton = newton(suite, -2.0, DEFAULT_TOL, None, None);
    let neg_secant = secant(suite.f, -3.0, -2.0, DEFAULT_TOL, None, None);
    let neg_hybrid = hybrid(suite, -3.0, -2.0, DEFAULT_TOL, None);
    let neg_super_halley = super_halley(suite, -2.0, DEFAULT_TOL, None, None);

    write_bracket_history("exp1_neg_bisection", &neg_bisection)?;
    write_iterate_history("exp1_neg_newton", &neg_newton)?;
    write_iterate_history("exp1_neg_secant", &neg_secant)?;
    write_bracket_history("exp1_neg_hybrid", &neg_hybrid)?;
    write_iterate_history("exp1_neg_super_halley", &neg_super_halley)?;

    let alpha = Some(2.5);
    let double_newton = newton(suite, 2.0, DEFAULT_TOL, alpha, None);
    let double_secant = secant(suite.f, 2.0, 2.1, DEFAULT_TOL, alpha, None);
    let double_super_halley = super_halley(suite, 2.0, DEFAULT_TOL, alpha, None);

    write_iterate_history("exp1_double_newton", &double_newton)?;
    write_iterate_history("exp1_double_secant", &double_secant)?;
    write_iterate_history("exp1_double_super_halley", &double_super_halley)?;

    let (n, x, residual) = final_bracket_summary(&neg_bisection, suite.f);
    println!(
        "Experiment 1, bisection on [-3,-2]: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}"
    );

    for (name, result) in [
        ("Newton", &neg_newton),
        ("Secant", &neg_secant),
        (
            "Hybrid",
            &IterateResult {
                rows: neg_hybrid
                    .rows
                    .iter()
                    .map(|row| IterateRow {
                        n: row.n,
                        step: None,
                        fx_abs: suite.f(row.x).abs(),
                        x: row.x,
                        error: None,
                        rate: None,
                    })
                    .collect(),
                status: neg_hybrid.status.clone(),
            },
        ),
        ("Super Halley", &neg_super_halley),
    ] {
        let (n, x, residual, _) = final_iterate_summary(result);
        println!("Experiment 1, {name}: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}");
    }

    for (name, result) in [
        ("Newton", &double_newton),
        ("Secant", &double_secant),
        ("Super Halley", &double_super_halley),
    ] {
        let (n, x, residual, rate) = final_iterate_summary(result);
        let rate_str = rate
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| String::from("--"));
        println!(
            "Experiment 1, double root, {name}: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}, rate = {rate_str}"
        );
    }

    Ok(())
}

fn run_experiment2() -> io::Result<()> {
    let suite = experiment2_suite();
    let alpha = Some(f64::ln(2.0));

    write_function_plot("exp2_function", -1.0, 2.0, suite.f);

    let chord_result = chord(suite, 0.0, DEFAULT_TOL, alpha, Some(20));
    let secant_result = secant(suite.f, 0.0, 2.0, DEFAULT_TOL, alpha, Some(20));
    let newton_result = newton(suite, 0.0, DEFAULT_TOL, alpha, Some(20));
    let super_halley_result = super_halley(suite, 0.0, DEFAULT_TOL, alpha, Some(20));

    write_iterate_history("exp2_chord", &chord_result)?;
    write_iterate_history("exp2_secant", &secant_result)?;
    write_iterate_history("exp2_newton", &newton_result)?;
    write_iterate_history("exp2_super_halley", &super_halley_result)?;

    for (name, result) in [
        ("Chord", &chord_result),
        ("Secant", &secant_result),
        ("Newton", &newton_result),
        ("Super Halley", &super_halley_result),
    ] {
        let (n, x, residual, rate) = final_iterate_summary(result);
        let rate_str = rate
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| String::from("--"));
        println!(
            "Experiment 2, {name}: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}, rate = {rate_str}"
        );
    }

    Ok(())
}

fn run_experiment3() -> io::Result<()> {
    let suite = experiment3_suite();
    let alpha = Some(0.0);

    write_function_plot("exp3_function", -2.0, 2.0, suite.f);

    let newton_pos = newton(suite, 1.0, EXP3_TOL, alpha, Some(30));
    let newton_near_singular = newton(suite, -0.705, EXP3_TOL, alpha, Some(30));
    let hybrid_bracket1 = hybrid(suite, -0.705, 1.0, EXP3_TOL, alpha);
    let hybrid_bracket2 = hybrid(suite, -0.71, 2.71, EXP3_TOL, alpha);

    write_iterate_history("exp3_newton_pos", &newton_pos)?;
    write_iterate_history("exp3_newton_near_singular", &newton_near_singular)?;
    write_bracket_history("exp3_hybrid_bracket1", &hybrid_bracket1)?;
    write_bracket_history("exp3_hybrid_bracket2", &hybrid_bracket2)?;

    for (name, result) in [
        ("Newton from x0=1", &newton_pos),
        ("Newton from x0=-0.705", &newton_near_singular),
    ] {
        let (n, x, residual, rate) = final_iterate_summary(result);
        let rate_str = rate
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| String::from("--"));
        println!(
            "Experiment 3, {name}: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}, rate = {rate_str}"
        );
    }

    for (name, result) in [
        ("Hybrid on [-0.705,1]", &hybrid_bracket1),
        ("Hybrid on [-0.71,2.71]", &hybrid_bracket2),
    ] {
        let (n, x, residual) = final_bracket_summary(result, suite.f);
        println!("Experiment 3, {name}: N = {n}, x_N = {x:.10e}, |f(x_N)| = {residual:.10e}");
    }

    Ok(())
}

fn run_grad_section() -> io::Result<()> {
    let scan_rows: Vec<GradScanRow> = (0..=16)
        .map(|i| {
            let a = 1.92 + i as f64 * 2.0e-6;
            let b = 2.08 - i as f64 * 1.0e-6;
            let result = bisection(poly9, a, b, GRAD_TOL, Some(2.0));
            let last = result.rows.last().expect("missing scan row");
            GradScanRow {
                a,
                b,
                root: last.x,
                iterations: last.n,
                residual: poly9(last.x).abs(),
            }
        })
        .collect();
    write_grad_scan(&scan_rows);

    let safe_result = safe_bisection_with_horner(&POLY9_COEFFS, 1.0, 2.5, GRAD_TOL, 2.0);
    write_safe_history("grad_safe", &safe_result)?;

    if let Some(last) = safe_result.rows.last() {
        println!(
            "Grad section, safeguarded bisection: N = {}, x_N = {:.10e}, |p(x_N)| = {:.10e}",
            last.n,
            last.x,
            poly9(last.x).abs()
        );
    }

    Ok(())
}

pub fn generate() -> io::Result<()> {
    run_experiment1()?;
    run_experiment2()?;
    run_experiment3()?;
    run_grad_section()?;

    util::plot("ch3")?;
    util::run_python_script("scripts/ch3/make_tables.py")?;

    let figure_pairs = [
        (
            "plots/ch3/exp1_function.png",
            "reports/ch3/figures/exp1_function.png",
        ),
        (
            "plots/ch3/exp1_multiple_root.png",
            "reports/ch3/figures/exp1_multiple_root.png",
        ),
        (
            "plots/ch3/exp2_function.png",
            "reports/ch3/figures/exp2_function.png",
        ),
        (
            "plots/ch3/exp2_rates.png",
            "reports/ch3/figures/exp2_rates.png",
        ),
        (
            "plots/ch3/exp3_function.png",
            "reports/ch3/figures/exp3_function.png",
        ),
        (
            "plots/ch3/exp3_locality.png",
            "reports/ch3/figures/exp3_locality.png",
        ),
        (
            "plots/ch3/grad_instability.png",
            "reports/ch3/figures/grad_instability.png",
        ),
    ];

    for (src, dst) in figure_pairs {
        util::copy_file(src, dst)?;
    }

    util::build_report("reports/ch3", "3.pdf")
}

#[cfg(test)]
mod tests {
    use super::{
        FunctionSuite, bisection, chord, newton, safe_bisection_with_horner, secant, super_halley,
    };
    use std::f64::consts::{LN_2, SQRT_2};

    fn test_f(x: f64) -> f64 {
        x * x - 2.0
    }

    fn test_df(x: f64) -> f64 {
        2.0 * x
    }

    fn test_ddf(_: f64) -> f64 {
        2.0
    }

    fn exp_f(x: f64) -> f64 {
        2.0 - x.exp()
    }

    fn exp_df(x: f64) -> f64 {
        -x.exp()
    }

    fn exp_ddf(x: f64) -> f64 {
        -x.exp()
    }

    #[test]
    fn bisection_finds_sqrt_two() {
        let result = bisection(test_f, 0.0, 2.0, 1.0e-12, Some(SQRT_2));
        let last = result.rows.last().expect("missing row");
        assert!((last.x - SQRT_2).abs() < 1.0e-12);
    }

    #[test]
    fn newton_converges_and_chord_stays_defined_on_two_minus_exp_x() {
        let suite = FunctionSuite {
            f: exp_f,
            df: exp_df,
            ddf: exp_ddf,
        };

        let newton_result = newton(suite, 0.0, 1.0e-12, Some(LN_2), None);
        let chord_result = chord(suite, 0.0, 1.0e-12, Some(LN_2), Some(20));
        let newton_last = newton_result.rows.last().expect("missing Newton row");
        let chord_last = chord_result.rows.last().expect("missing chord row");

        assert!((newton_last.x - LN_2).abs() < 1.0e-12);
        assert_eq!(chord_last.n, 20);
        assert!(chord_last.x.is_finite());
    }

    #[test]
    fn secant_and_super_halley_converge_on_sqrt_two() {
        let suite = FunctionSuite {
            f: test_f,
            df: test_df,
            ddf: test_ddf,
        };

        let secant_result = secant(test_f, 1.0, 2.0, 1.0e-12, Some(SQRT_2), None);
        let halley_result = super_halley(suite, 2.0, 1.0e-12, Some(SQRT_2), None);
        let secant_last = secant_result.rows.last().expect("missing secant row");
        let halley_last = halley_result.rows.last().expect("missing Halley row");

        assert!((secant_last.x - SQRT_2).abs() < 1.0e-10);
        assert!((halley_last.x - SQRT_2).abs() < 1.0e-12);
    }

    #[test]
    fn safeguarded_bisection_detects_roundoff_issue() {
        let coeffs = [
            1.0, -18.0, 144.0, -672.0, 2016.0, -4032.0, 5376.0, -4608.0, 2304.0, -512.0,
        ];
        let result = safe_bisection_with_horner(&coeffs, 1.0, 2.5, 1.0e-18, 2.0);
        assert!(result.status.contains("roundoff"));
        assert!(!result.rows.is_empty());
    }
}
