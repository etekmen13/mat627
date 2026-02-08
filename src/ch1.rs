use crate::util;
use std::io;
#[allow(dead_code)]
fn alternating(x: i32, n: i32) -> f64 {
    let x = x as f64;
    let mut term: f64 = 1.0;
    let mut sum: f64 = 1.0;
    for i in 1..=n {
        term *= (-x) / (i as f64);
        sum += term;
    }

    sum
}
#[allow(dead_code)]
fn reciprocal(x: i32, n: i32) -> f64 {
    let x = x as f64;
    let mut term: f64 = 1.0;
    let mut sum: f64 = 1.0;
    for i in 1..=n {
        term *= x / (i as f64);
        sum += term;
    }

    1.0 / sum
}

#[derive(PartialEq)]
pub enum ApproximationType {
    Alternating,
    Reciprocal,
}
#[must_use]
pub fn test_p1(approx_type: ApproximationType) {
    let (name, func) = match approx_type {
        ApproximationType::Alternating => ("Alternating", alternating as fn(i32, i32) -> f64),
        ApproximationType::Reciprocal => ("Reciprocal", reciprocal as fn(i32, i32) -> f64),
    };
    println!(
        "Testing {} Series Approximation for e^{{-x}} for x∈{{-50..1000}} and n∈{{1..100}}...",
        name
    );
    let x_list: [i32; 13] = [-50, -20, -15, -10, -5, -1, 1, 5, 10, 50, 100, 500, 1000];
    let n_list: [i32; 100] = std::array::from_fn(|i| (i + 1) as i32);

    let _ = x_list.into_iter().map(|x| {
        let exact = f64::exp(-x as f64);
        let data: [f64; 100] = std::array::from_fn(|j| {
            let n = n_list[j];
            let approx = func(x, n);
            util::rel_error(approx, exact)
        });
        util::write_data(&data, format!("data/ch1/{}", name), x.to_string());
    });
}

fn find_smallest_unrepresentable_n() -> u64 {
    let mut l: u64 = 0;
    let mut r: u64 = u64::MAX;

    while l < r {
        let mid: u64 = l + (r - l) / 2u64;
        if (mid as f32) as u64 != mid {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    l
}

pub fn test_p2() -> io::Result<()> {
    let n: u64 = find_smallest_unrepresentable_n();
    println!(
        "Smallest unrepresentable n as float32: \x1b[32m{}\x1b[0m",
        n
    );

    let vals: Vec<String> = (n - 2..n + 4)
        .map(|x| {
            if (x as f32) as u64 != x {
                format!("\x1b[31m{}\x1b[0m", (x as f32))
            } else {
                format!("\x1b[32m{}\x1b[0m", (x as f32))
            }
        })
        .collect();

    println!("Proof:\n");
    for v in &vals {
        print!("\t{v}");
    }
    println!();
    Ok(())
}
