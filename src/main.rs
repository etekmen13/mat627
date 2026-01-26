mod ch1;
use std::env;
use std::{fs, io};
fn main() {
    let args: Vec<String> = env::args().collect();

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
    let ha = ch1::test_p1(ch1::ApproximationType::Alternating);
    let hr = ch1::test_p1(ch1::ApproximationType::Reciprocal);

    for handle in ha {
        handle.join().expect("thread panicked");
    }
    for handle in hr {
        handle.join().expect("thread panicked");
    }
    println!("Done.");

    println!("Plotting data...");

    ch1::plot_p1().expect("plot error");

    println!("Done.");
}

fn ch1p2() {
    println!("\n=== Chapter 1 Problem 2 ===");

    ch1::test_p2().expect("error");
}
