use clap::{command, error::ErrorKind, CommandFactory, Parser, ValueEnum};
use cudarc::{
    driver::{
        sys::*, CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    },
    nvrtc::Ptx,
};
use host_code::CudaEvent;
use num::traits::float::FloatCore;
use rayon::{iter::repeatn, prelude::*};
use std::{
    any::TypeId,
    fmt::Display,
    io::{self, Write},
    iter::{once, successors},
    sync::Arc,
};
/// Reduction on Nvidia GPU written in Rust
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Benchmark runs for input sizes of 2^(from), 2^(from+inc), ... , 2^(to)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=31))]
    from: u32,
    /// Benchmark runs for input sizes of 2^(from), 2^(from+inc), ... , 2^(to)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=31))]
    to: Option<u32>,
    /// Benchmark runs for input sizes of 2^(from), 2^(from+inc), ... , 2^(to)
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    inc: u32,
    /// Benchmark runs for block-sizes from [dim_min, dim_min * 2, ... , dim_max]
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(32..=1024))]
    dim_min: u32,
    /// Benchmark runs for block-sizes from [dim_min, dim_min * 2, ... , dim_max]
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(32..=1024))]
    dim_max: u32,
    /// Minimum number of thread iterations in 2^thread_iter_min
    /// (A limit of 1024 blocks total per grid is always respected)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    thread_iter_min: Option<u32>,
    /// Maximum number of thread iterations in 2^thread_iter_max
    /// (A limit of 1024 blocks total per grid is always respected)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    thread_iter_max: Option<u32>,
    /// Value type
    #[arg(long = "type", value_enum, default_value_t = ValueType::Both)]
    value_type: ValueType,
    /// Benchmark runs for a repititions number of times per parameter set
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    reps: u32,
    /// Disable cyclic allocation on the device (helpful if only the kernel duration
    /// is of interest)
    #[arg(long, default_value_t = false, conflicts_with("check"))]
    no_cyclic_alloc: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ValueType {
    /// f32 (float)
    F32,
    /// f64 (double)
    F64,
    /// f32 + f64 (float + double)
    Both,
}

// Configuration to execute the kernel
struct ExecConfig<'a> {
    block_dims: &'a [u32],
    min_thread_iterations: u32,
    max_thread_iterations: u32,
    reps: u32,
    no_cyclic_alloc: bool,
}

// Helper function to execute the kernel depending on type, configuration etc.
fn exec_reduction<T>(
    input: &[T],
    exec_cfg: &ExecConfig,
    result: T,
    device: &Arc<CudaDevice>,
) -> Result<(), DriverError>
where
    T: FloatCore + 'static + DeviceRepr + ValidAsZeroBits + Default + Unpin + Display,
{
    // We lock the stdout only one time per function invocation to improve performance
    let mut stdout = io::stdout().lock();
    // We must choose the right kernel depending on type of T
    let (kernel, kernel_name) = if TypeId::of::<T>() == TypeId::of::<f32>() {
        (
            device.get_func("kernels", "reduction_f32").unwrap(),
            "reduction_f32",
        )
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        (
            device.get_func("kernels", "reduction_f64").unwrap(),
            "reduction_f64",
        )
    } else {
        unreachable!()
    };
    // get input size
    let n = input.len();
    // Create events to measure times
    let mut pre_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut pre_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    // if cyclic allocation is disabled we allocate the device buffers here already
    let mut device_buffers = if exec_cfg.no_cyclic_alloc {
        Some((device.htod_sync_copy(input)?, device.alloc_zeros::<T>(1)?))
    } else {
        None
    };
    // Iterate over the given block sizes
    for &block_dim in exec_cfg.block_dims {
        // Calculate grid dimensions, make sure max_grid is always included
        let min_grid = ((n as u32 + block_dim * (1 << exec_cfg.max_thread_iterations) - 1)
            / (block_dim * (1 << exec_cfg.max_thread_iterations)))
            .min(1024);
        let max_grid = ((n as u32 + block_dim * (1 << exec_cfg.min_thread_iterations) - 1)
            / (block_dim * (1 << exec_cfg.min_thread_iterations)))
            .min(1024);
        let grid_dims = successors(Some(min_grid), |&dim| Some(dim << 1))
            .take_while(|&dim| dim < max_grid)
            .chain(once(max_grid));
        // Iterate over grid dimensions
        for grid_dim in grid_dims {
            // Create the launch configuration, shared memory must be large enough to hold
            // one element per thread
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: block_dim * size_of::<T>() as u32,
            };
            // Iterate over the given number of repititions
            for i in 0..exec_cfg.reps {
                // If device input and device output are already allocated, we skip cyclic allocation
                let kernel_result = if let Some((d_input, d_output)) = &mut device_buffers {
                    // Record event before calling into CUDA
                    pre_event.record()?;
                    // Record event before kernel invocation
                    pre_kernel_event.record()?;
                    unsafe { kernel.clone().launch(cfg, (d_input, d_output, n))? }
                    // Record event after kernel invocation
                    post_kernel_event.record()?;
                    // Record event after calling into CUDA and sync the device
                    post_event.record()?;
                    post_event.sync()?;
                    None
                } else {
                    // Record event before calling into CUDA
                    pre_event.record()?;
                    // Allocate memory on the GPU and copy input to it
                    let d_input = device.htod_sync_copy(input)?;
                    let mut d_output = device.alloc_zeros::<T>(1)?;
                    // Record event before kernel invocation
                    pre_kernel_event.record()?;
                    unsafe { kernel.clone().launch(cfg, (&d_input, &mut d_output, n))? }
                    // Record event after kernel invocation
                    post_kernel_event.record()?;
                    // Copy back result and drop the device input buffer
                    let kernel_result = device.sync_reclaim(d_output)?[0];
                    drop(d_input);
                    // Record event after calling into CUDA and sync the device
                    post_event.record()?;
                    post_event.sync()?;
                    Some(kernel_result)
                };
                // Calculate elapsed durations
                let (pre_dur, kernel_dur, post_dur) = (
                    pre_event.elapsed(&pre_kernel_event)?,
                    pre_kernel_event.elapsed(&post_kernel_event)?,
                    post_kernel_event.elapsed(&post_event)?,
                );
                // Write to stdout the result and measurements of the current invocation
                let output = if let Some(kernel_result) = kernel_result {
                    format!(
                    "{kernel_name};{i};{n};{block_dim};{grid_dim};{pre_dur};{kernel_dur};{post_dur};{:.0};{:.0}\n",kernel_result,result
                )
                } else {
                    format!("{kernel_name};{i};{n};{block_dim};{grid_dim};{kernel_dur}\n")
                };
                stdout.write_all(output.as_bytes()).unwrap();
            }
        }
    }
    Ok(())
}

fn main() {
    // Parse the given command line arguments and validate them
    let Cli {
        from,
        to,
        inc,
        dim_min,
        dim_max,
        reps,
        thread_iter_min,
        thread_iter_max,
        value_type,
        no_cyclic_alloc,
    } = Cli::parse();
    let to = to.unwrap_or(from);
    if to < from {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "to must be greater or equal from",
            )
            .exit();
    }
    if dim_max < dim_min || !dim_min.is_power_of_two() || !dim_max.is_power_of_two() {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "block_dim_max must be greater or equal block_dim_min and both must be a power of two",
            )
            .exit();
    }
    let (min_thread_iterations, max_thread_iterations) = match (thread_iter_min, thread_iter_max) {
        (None, None) => (1, 1),
        (None, Some(max)) => (1, max),
        (Some(min), None) => (min, min),
        (Some(min), Some(max)) => {
            if max < min {
                Cli::command()
                    .error(
                        ErrorKind::ArgumentConflict,
                        "min_thread_iterations must be smaller or equal max_thread_iterations",
                    )
                    .exit();
            }
            (min, max)
        }
    };
    // Calculate the input sizes to iterate on, make sure 2^to is always included
    let input_sizes = successors(Some(from), |&val| Some(val + inc))
        .take_while(|&val| val < to)
        .chain(once(to));
    // Calculate the block_dims to iterate on, make sure block_dim_max is always included
    let block_dims: Vec<u32> = successors(Some(dim_min), |&dim| Some(dim << 1))
        .take_while(|&dim| dim < dim_max)
        .chain(once(dim_max))
        .collect();
    // Create the execution configuration
    let exec_config = ExecConfig {
        block_dims: &block_dims,
        min_thread_iterations,
        max_thread_iterations,
        reps,
        no_cyclic_alloc,
    };
    // Create the Cuda Device
    let dev = CudaDevice::new(0).unwrap();
    // Load the kernel file and compile it
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["reduction_f32", "reduction_f64"],
    )
    .unwrap();
    // Print the header
    if !no_cyclic_alloc {
        println!("kernel;iteration;input-size;block-dim;grid-dim;pre-duration;kernel-duration;post-duration;kernel-result;cpu-result");
    } else {
        println!("kernel;iteration;input-size;block-dim;grid-dim;kernel-duration");
    }
    // Iterate over the input size
    for i in input_sizes {
        // Call helper function for f32 type if necessary
        if value_type == ValueType::F32 || value_type == ValueType::Both {
            // Generate input vec on host and calculate the sum (we use rayon to make this a bit faster)
            let input: Vec<f32> = repeatn(1f32, 1usize << i).collect();
            let result = input.par_iter().sum();
            // Call helper function
            exec_reduction(&input, &exec_config, result, &dev).unwrap();
        }
        // Call helper function for f64 type if necessary
        if value_type == ValueType::F64 || value_type == ValueType::Both {
            // Generate input vec on host and calculate the sum (we use rayon to make this a bit faster)
            let input: Vec<f64> = repeatn(1f64, 1usize << i).collect();
            let result = input.par_iter().sum();
            // Call helper function
            exec_reduction(&input, &exec_config, result, &dev).unwrap();
        }
    }
}
