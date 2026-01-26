use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=scripts/");
    println!("cargo:rerun-if-changed=pyproject.toml");

    let uv_check = Command::new("uv").arg("--version").output();

    if uv_check.is_err() {
        eprintln!(
            "Warning: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        );
        eprintln!("Skipping Python dependency installation.");
        return;
    }

    let status = Command::new("uv")
        .args(&["sync", "--frozen"])
        .status()
        .expect("Failed to run uv sync");

    if !status.success() {
        panic!("Failed to install Python dependencies");
    }
}
