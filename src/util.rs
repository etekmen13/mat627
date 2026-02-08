use npy_writer::NumpyWriter;
use std::fs;
use std::io;
use std::process::{Command, Stdio};

/// Write an array of f64 to an .npy file
///
/// # Arguments
/// * `data` - The array of f64s
/// * `dir` - The directory to write to
/// * `fname` - the filename
pub fn write_data(data: &[f64], dir: String, fname: String) {
    fs::create_dir_all(&dir).unwrap();

    let full_path = format!("{}/{}.npy", &dir, &fname);

    let mut f = fs::File::create(&full_path).unwrap();

    data.write_npy(&mut f).unwrap();
}
pub fn rel_error(approx: f64, exact: f64) -> f64 {
    f64::abs(approx - exact) / f64::abs(exact)
}

/// Plots data according to the plot.py script.
/// Assumes plot.py in scripts/{chapter}
/// Assumes data in data/{chapter}
/// Places plot  in plots/{chapter}
pub fn plot(chapter: &str) -> io::Result<()> {
    let status = Command::new("python")
        .arg(format!("scripts/{}/plot.py", chapter))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("plot failed: {status}"),
        ));
    }
    Ok(())
}
