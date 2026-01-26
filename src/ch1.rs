use npy_writer::NumpyWriter;
use std::fs;
use std::io;
use std::process::Command;
use std::thread;

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

fn write_data(data: &[f64], dir: String, fname: String) {
    fs::create_dir_all(&dir).unwrap();

    let full_path = format!("{}/{}.npy", &dir, &fname);

    let mut f = fs::File::create(&full_path).unwrap();

    data.write_npy(&mut f).unwrap();
}

fn rel_error(approx: f64, exact: f64) -> f64 {
    f64::abs(approx - exact) / f64::abs(exact)
}

fn compute(x: i32, f: fn(i32, i32) -> f64) -> [f64; 100] {
    let n_list: [i32; 100] = std::array::from_fn(|i| (i + 1) as i32);

    let mut out_list: [f64; 100] = [0.0; 100];
    for i in 0..100 {
        let approx = f(x, n_list[i]);
        let exact = f64::exp(-x as f64);

        out_list[i] = rel_error(approx, exact);
    }
    out_list
}

#[derive(PartialEq)]
pub enum ApproximationType {
    Alternating,
    Reciprocal,
}
#[must_use]
pub fn test_p1(approx_type: ApproximationType) -> Vec<thread::JoinHandle<()>> {
    let (name, func) = match approx_type {
        ApproximationType::Alternating => ("Alternating", alternating as fn(i32, i32) -> f64),
        ApproximationType::Reciprocal => ("Reciprocal", reciprocal as fn(i32, i32) -> f64),
    };
    println!(
        "Testing {} Series Approximation for e^{{-x}} for x∈{{-50..1000}} and n∈{{1..100}}...",
        name
    );
    let x_list: [i32; 13] = [-50, -20, -15, -10, -5, -1, 1, 5, 10, 50, 100, 500, 1000];

    let mut handles = vec![];
    for x in x_list {
        let handle = thread::spawn(move || {
            let data: [f64; 100] = compute(x, func);
            write_data(&data, format!("data/ch1/{}", name), x.to_string())
        });
        handles.push(handle);
    }
    handles
}

pub fn plot_p1() -> io::Result<()> {
    let output = Command::new("python")
        .arg("scripts/ch1/plot.py")
        .arg("data/ch1")
        .arg(format!("plots/ch1/plot.png"))
        .output()?;

    if !output.status.success() {
        eprintln!("Python error: {}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
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
