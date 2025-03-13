use std::env;
use std::fs::canonicalize;
use std::process::{Command, Stdio};

type DynError = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{}", e);
        std::process::exit(-1);
    }
}

fn try_main() -> Result<(), DynError> {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("build") => build()?,
        Some("run") => run()?,
        _ => print_help(),
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "Tasks:

build           Builds kernels with nightly toolchain and host-code with given arguments 
run             Runs host-code with given arguments 
"
    )
}

fn build() -> Result<(), DynError> {
    build_kernels()?;

    let args: Vec<String> = env::args().collect();
    let mut command = Command::new("cargo")
        .current_dir(canonicalize("./host-code")?)
        .arg("build")
        .args(args.into_iter().skip(2))
        .stdout(Stdio::piped())
        .spawn()?;

    let output = command.wait()?;
    if !output.success() {
        Err("Building host-code failed")?;
    };

    Ok(())
}

fn run() -> Result<(), DynError> {
    build_kernels()?;

    let args: Vec<String> = env::args().collect();
    let mut command = Command::new("cargo")
        .current_dir(canonicalize("./host-code").unwrap())
        .arg("run")
        .args(args.into_iter().skip(2))
        .stdout(Stdio::piped())
        .spawn()?;

    let output = command.wait()?;
    if !output.success() {
        Err("Building host-code failed")?;
    };

    Ok(())
}

fn build_kernels() -> Result<(), DynError> {
    let capability = cuda_device_capability()?;

    let mut command = Command::new("cargo")
        .current_dir(canonicalize("./kernels")?)
        .arg("+nightly")
        .arg("rustc")
        .arg("--release")
        .arg("--")
        .arg("-C")
        .arg(format!("target-cpu=sm_{capability}"))
        .arg("-Zmir-enable-passes=-JumpThreading")
        .stdout(Stdio::piped())
        .spawn()?;

    let output = command.wait()?;
    if !output.success() {
        Err("Building kernels for device failed")?;
    }
    Ok(())
}

fn cuda_device_capability() -> Result<i32, DynError> {
    let capability = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()?;
    let capability = String::from_utf8(capability.stdout)?
        .lines()
        .next()
        .unwrap()
        .trim_end()
        .parse::<f64>()?;
    Ok((capability * 10.) as i32)
}
