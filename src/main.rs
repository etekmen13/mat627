mod ch1;
mod ch2_1;
mod ch2_2;
mod util;
use std::env;
use std::{fs, io};
fn main() {
    let args: Vec<String> = env::args().collect();
    // let [a, b, c] = ch2_1::magnitudes(1.92, 2.08);
    // println!("{:.4e} {:.4e} {:.4e}", a, b, c);
    if args.len() < 2 {
        eprintln!("Please provide the chapter/section number (e.g. '1', '2.1', '2.2')");
        return;
    }
    match args[1].as_str() {
        "1" => {
            make_dirs("ch1").expect("Error making directories.");
            ch1p1();
            ch1p2();
        }
        "2.1" => {
            make_dirs("ch2_1").expect("Error making directories.");
            ch2_1();
        }

        _ => println!("Chapter/section unrecognized."),
    }
}

fn make_dirs(name: &str) -> io::Result<()> {
    let dirs: [&'static str; 4] = ["scripts", "reports", "plots", "data"];
    for d in dirs {
        let path = format!("{}/{}", d, name);
        fs::create_dir_all(path)?;
    }

    Ok(())
}
fn ch1p1() {
    println!("\n=== Chapter 1 Problem 1 ===");
    let _ = ch1::test_p1(ch1::ApproximationType::Alternating);
    let _ = ch1::test_p1(ch1::ApproximationType::Reciprocal);
    println!("Done.");

    println!("Plotting data...");

    util::plot("ch1").expect("plot error");

    println!("Done.");
}

fn ch1p2() {
    println!("\n=== Chapter 1 Problem 2 ===");

    ch1::test_p2().expect("error");
}

fn ch2_1() {
    let a = 1.92;
    let b = 2.08;
    println!("Polynomial p(x) = (x-2)^9 on the interval [1.92, 2.08]");
    println!("Comparing Standard vs. Horners at N=1000...\n");
    {
        let [standard, horners] = ch2_1::compare_methods::<1000>(a, b);

        println!("Standard Evaluation Max Absolute Error: {:.4e}", standard);
        println!("Horners Evaluation Max Absolute Error:  {:.4e}", horners);
    }

    println!("\nComparing Standard vs. Horners at N=100,000...\n");
    {
        let [standard, horners] = ch2_1::compare_methods::<100_000>(a, b);

        println!("Standard Evaluation Max Absolute Error: {:.4e}", standard);
        println!("Horners Evaluation Max Absolute Error:  {:.4e}", horners);
    }

    println!("\n(Grad) Plotting Exact, Standard, and Horners Evalutations at N=100_000...");
    ch2_1::plot_methods::<100_000>(a, b).expect("Error plotting");
}
