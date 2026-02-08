use crate::util;
use std::io;
fn p_standard(x: f64, coeff: &[f64]) -> f64 {
    let mut term: f64 = 1.0;
    coeff.iter().fold(0.0, |acc, &a| {
        let s = acc + a * term;
        term *= x;
        s
    })
}
fn p_horners(x: f64, coeff: &[f64]) -> f64 {
    coeff.iter().rev().fold(0.0, |acc, &a| acc.mul_add(x, a))
}
/// Generate N samples from [a,b] according to D = {a + kh} where h = (b-a)/N
///
/// # Arguments
/// `a` - lower bound (inclusive)
/// `b` - upper bound (inclusive)
fn domain<const N: usize>(a: f64, b: f64) -> [f64; N] {
    let h: f64 = (b - a) / (N as f64);
    std::array::from_fn(|k| a + (k as f64) * h)
}

pub fn compare_methods<const N: usize>(lower: f64, upper: f64) -> [f64; 2] {
    let coeff = vec![
        -512.0, 2304.0, -4608.0, 5376.0, -4032.0, 2016.0, -672.0, 144.0, -18.0, 1.0,
    ];
    let d = domain::<N>(lower, upper);
    let standard: f64 = d
        .into_iter()
        .reduce(|a, b| {
            let exact_a = f64::powi(a - 2.0, 9);
            let exact_b = f64::powi(b - 2.0, 9);
            let diff_a = f64::abs(exact_a - p_standard(a, &coeff));
            let diff_b = f64::abs(exact_b - p_standard(b, &coeff));
            diff_a.max(diff_b)
        })
        .unwrap_or(f64::NEG_INFINITY);
    let horners: f64 = d
        .into_iter()
        .reduce(|a, b| {
            let exact_a = f64::powi(a - 2.0, 9);
            let exact_b = f64::powi(b - 2.0, 9);
            let diff_a = f64::abs(exact_a - p_horners(a, &coeff));
            let diff_b = f64::abs(exact_b - p_horners(b, &coeff));
            diff_a.max(diff_b)
        })
        .unwrap_or(f64::NEG_INFINITY);
    [standard, horners]
}

pub fn plot_methods<const N: usize>(lower: f64, upper: f64) -> io::Result<()> {
    let d = domain::<N>(lower, upper);

    let coeff = vec![
        -512.0, 2304.0, -4608.0, 5376.0, -4032.0, 2016.0, -672.0, 144.0, -18.0, 1.0,
    ];
    let exact: [f64; N] = std::array::from_fn(|i| f64::powi(d[i] - 2.0, 9));
    let standard: [f64; N] = std::array::from_fn(|i| p_standard(d[i], &coeff));
    let horners: [f64; N] = std::array::from_fn(|i| p_horners(d[i], &coeff));

    let data_path = "data/ch2_1";

    util::write_data(&exact, data_path.to_string(), String::from("exact"));
    util::write_data(&standard, data_path.to_string(), String::from("standard"));
    util::write_data(&horners, data_path.to_string(), String::from("horners"));

    util::plot("ch2_1")
}
