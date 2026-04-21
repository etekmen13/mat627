use npy_writer::NumpyWriter;
use std::fs;
use std::io;
use std::path::Path;
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
    run_python_script(&format!("scripts/{}/plot.py", chapter))
}

pub fn run_python_script(path: &str) -> io::Result<()> {
    if command_exists("uv") && Path::new("pyproject.toml").exists() {
        let mut cmd = Command::new("uv");
        cmd.env("UV_CACHE_DIR", "/tmp/uv-cache");
        cmd.env("MPLCONFIGDIR", "/tmp/matplotlib");
        cmd.arg("run").arg("python").arg(path);
        run_command(&mut cmd, &format!("python script failed: {path}"))
    } else {
        let mut cmd = Command::new(python_executable()?);
        cmd.env("MPLCONFIGDIR", "/tmp/matplotlib");
        cmd.arg(path);
        run_command(&mut cmd, &format!("python script failed: {path}"))
    }
}

pub fn copy_file(src: &str, dst: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(dst).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(src, dst)?;
    Ok(())
}

pub fn build_report(report_dir: &str, output_name: &str) -> io::Result<()> {
    let mut cmd = Command::new("latexmk");
    cmd.arg("-r")
        .arg(".latexmkrc")
        .arg("-pdf")
        .arg("main.tex")
        .current_dir(report_dir);

    run_command(&mut cmd, &format!("latex build failed: {report_dir}"))?;
    copy_file(
        &format!("{report_dir}/build/main.pdf"),
        &format!("{report_dir}/{output_name}"),
    )
}

fn run_command(command: &mut Command, message: &str) -> io::Result<()> {
    let status = command
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("{message}: {status}")))
    }
}

fn python_executable() -> io::Result<&'static str> {
    if command_exists("python") {
        Ok("python")
    } else if command_exists("python3") {
        Ok("python3")
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "neither 'python' nor 'python3' is available",
        ))
    }
}

fn command_exists(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}
