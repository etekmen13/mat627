use crate::util;
use std::io;
use std::iter;

fn forward_diff(x: f64, h: f64, f: fn(f64) -> f64) -> f64 {
    (f(x + h) - f(x)) / h
}

fn backward_diff(x: f64, h: f64, f: fn(f64) -> f64) -> f64 {
    (f(x) - f(x - h)) / h
}

fn center_diff(x: f64, h: f64, f: fn(f64) -> f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn convergence_rate(err_curr: f64, err_next: f64) -> f64 {
    f64::ln((err_next / err_curr).abs()) / f64::ln(2.0)
}
fn compare(input: f64) {
    let fs = [|x: f64| (x + 1.0).sqrt(), |x: f64| f64::exp(x)];

    let dfs = [|x: f64| 0.5 / (x + 1.0).sqrt(), |x: f64| f64::exp(x)];

    let hs = iter::successors(Some(0.5), |&x| Some(x / 2.0));

    let ds = [forward_diff, backward_diff, center_diff];

    // for each f-df pair, for each approx, for each h, compute approx, error, and rate
    fs.iter().zip(dfs.iter()).flat_map(|&(f, df)| {
        ds.iter().flat_map(|&d| {
            hs.iter().map(|&h| {
                [
                    d(input, h, f),
                    df(input) - d(input, h, f),
                    convergence_rate(),
                ]
            })
        })
    })
}
