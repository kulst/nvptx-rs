use clap::{command, error::ErrorKind, CommandFactory, Parser, ValueEnum};
use cudarc::{
    driver::{
        sys::*, CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    },
    nvrtc::Ptx,
};
use host_code::CudaEvent;
use num::traits::float::FloatCore;
use num_traits::{Float, Zero};
use rayon::{iter::repeatn, prelude::*};
use std::{
    any::TypeId, fmt::Display, io::{self, Write}, iter::{successors, Once}, sync::Arc
};
use std::iter::Sum;
use std::iter::once;
use std::ops::AddAssign;

/// matrixmultiplication on Nvidia GPU written in Rust
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Benchmark runs for input sizes of 2^(from), 2^(from+increment), ... , 2^(to)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=14))]
    from: Option<u32>,
    /// Benchmark runs for input sizes of 2^(from), 2^(from+increment), ... , 2^(to)
    #[arg(long, default_value_t = 13, value_parser = clap::value_parser!(u32).range(1..=14))]
    to: u32,
    /// Benchmark runs for input sizes of 2^(from), 2^(from+increment), ... , 2^(to)
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=3))]
    increment: u32,
    #[arg(value_enum, default_value_t = ValueType::Both)]
    arg_type: ValueType,
    /// Use shared memory
    //#[arg(long, default_value_t = true)]
    //use_shared_memory: bool,
    /// Benchmark runs for a repititions number of times per parameter set
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..))]
    repititions: u32,
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
struct ExecConfig {
    repititions: u32,
}

// Helper function to execute the kernel depending on type, configuration etc.
fn exec_matrixmultiplication<T>(
    h_a: &[T],
    h_b: &[T],
    n: u32,
    exec_cfg: &ExecConfig,
    use_shared_memory: bool,
    device: &Arc<CudaDevice>,
) -> Result<(), DriverError>
where
    T: FloatCore + 'static + DeviceRepr + ValidAsZeroBits + Default + Unpin + Display + AddAssign,
{
    // We lock the stdout only one stime per function invocation to improve performance
    let mut stdout = io::stdout().lock();
    // We must choose the right kernel depending on type of T
    let (kernel, kernel_name) = if (TypeId::of::<T>() == TypeId::of::<f32>()) && use_shared_memory {
        (
            device.get_func("kernels", "matrixmultiplication_shared_memory_f32").unwrap(),
            "shared_f32",
        )
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        (
            device.get_func("kernels", "matrixmultiplication_f32").unwrap(),
            "simple_f32",
        )
    } else if (TypeId::of::<T>() == TypeId::of::<f64>()) && use_shared_memory {
        (
            device.get_func("kernels", "matrixmultiplication_shared_memory_f64").unwrap(),
            "shared_f64",
        )
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        (
            device.get_func("kernels", "matrixmultiplication_f64").unwrap(),
            "simple_f64",
        )
    } else {
        unreachable!()
    };
    // Create events to measure times
    let mut pre_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut pre_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
        
    // Create the launch configuration, shared memory must be large enough to hold
    // one element per thread+
    const TILE_SIZE: u32 = 16;
    let block_dim = TILE_SIZE;
    let grid_dim = (n+TILE_SIZE-1)/TILE_SIZE;
    let cfg = LaunchConfig {
        grid_dim: ((n+TILE_SIZE-1)/TILE_SIZE, (n+TILE_SIZE-1)/TILE_SIZE, 1),
        block_dim: (TILE_SIZE, TILE_SIZE, 1),
        shared_mem_bytes: 0,
    };
    // Iterate over the given number of repititions
    for i in 0..exec_cfg.repititions {
        // Record event before calling into C
        pre_event.record()?;
        // Allocate memory on the GPU and copy input to it
        let d_input_a = device.htod_sync_copy(h_a)?;
        let d_input_b = device.htod_sync_copy(h_b)?;
        let mut d_output = device.alloc_zeros::<T>((n*n) as usize)?;
                  
        // Record event before kernel invocation
        pre_kernel_event.record()?;
        unsafe { kernel.clone().launch(cfg, (&d_input_a, &d_input_b, &mut d_output, n as usize))? }
        // Record event after kernel invocation
        post_kernel_event.record()?;
        // Copy back result and drop the device input buffer
        let kernel_result = device.sync_reclaim(d_output)?;        	
        drop(d_input_a);
        drop(d_input_b);
        // Record event after calling into CUDA and sync the device
        post_event.record()?;
        post_event.sync()?;
        // Calculate elapsed durations
        let (pre_dur, kernel_dur, post_dur) = (
            pre_event.elapsed(&pre_kernel_event)?,
            pre_kernel_event.elapsed(&post_kernel_event)?,
            post_kernel_event.elapsed(&post_event)?,
        );
        
        let mut abs_diff_sum = T::zero(); // Start with zero
        for row in 0..n as usize{
            for col in 0..n as usize {
                abs_diff_sum += (h_a[row * n as usize + col] - kernel_result[row * n as usize + col]).abs();
            }
        }            
        
        // Write to stdout the result and measurements of the current invocation
        if abs_diff_sum < T::from(0.5).unwrap() {
            let n_squared = n*n;
            let total_dur: f64 = pre_dur as f64 + kernel_dur as f64 + post_dur as f64; 
            let mut throughput = 0.0;
            if(total_dur == 0.0){
                let throughput = 0.000001 * (n_squared as f64) / (total_dur as f64);
            }
            let output = format!(
                "{kernel_name};{i};{n};{n_squared};{pre_dur};{kernel_dur};{post_dur};{throughput}\n");
            stdout.write_all(output.as_bytes()).unwrap();
        }
    }
    Ok(())
}


fn initialize_matrix<T>(n: usize) -> (Vec<T>)
where
    T: Default + From<u16> + Copy + Display,
{
    let mut matrix: Vec<T> = vec![T::default(); n * n];

    for row in 0..n {
        for col in 0..n {
            matrix[row * n + col] = if row == col { T::from(1) } else { T::from(0) };
        }
    }

    matrix
}

fn main() {
    // Parse the given command line arguments and validate them
    let Cli {
        from,
        to,
        increment: step,
        arg_type,
        repititions,
    } = Cli::parse();
    let from = 10; //from.unwrap_or(to);
    if to < from {
        Cli::command()
        .error(
        ErrorKind::ArgumentConflict,
        "to must be greater or equal from",
        )
        .exit();
        }        
    // Calculate the input sizes to iterate on, make sure 2^to is always included
    let input_sizes: Vec<u32> = successors(Some(from), |&size| Some(size+step)).take_while(|&size| size < to).chain(once(to)).map(|exp| 1<<exp).collect();
    const TILE_SIZE: usize = 16;
    // Create the execution configuration
    let exec_config = ExecConfig {
        repititions,
    };
    // Create the Cuda Device
    let dev = CudaDevice::new(0).unwrap();
    // Load the kernel file and compile it
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["matrixmultiplication_f32", "matrixmultiplication_f64", "matrixmultiplication_shared_memory_f32", "matrixmultiplication_shared_memory_f64"],
    )
    .unwrap();
    
    // Print the header
    println!("kernel;iteration;n;n_squared;pre_dur;kernel_dur;post_dur;throughput;kernel_result");    // Iterate over the input size
    for n in input_sizes {
        
        // Call helper function for f32 type if necessary
        if arg_type == ValueType::F32 || arg_type == ValueType::Both {
            // Generate input vec on host and calculate the sum (we use rayon to make this a bit faster)
            let h_a: (Vec<f32>) = initialize_matrix(n as usize);
            let h_b: (Vec<f32>) = initialize_matrix(n as usize);
            // Call helper function
            exec_matrixmultiplication(&h_a, &h_b, n, &exec_config, true, &dev).unwrap();
            // Call helper function
            exec_matrixmultiplication(&h_a, &h_b, n, &exec_config, false, &dev).unwrap();
        }
        
        // Call helper function for f32 type if necessary
        if arg_type == ValueType::F64 || arg_type == ValueType::Both {
            // Generate input vec on host and calculate the sum (we use rayon to make this a bit faster)
            let h_a: (Vec<f64>) = initialize_matrix(n as usize);
            let h_b: (Vec<f64>) = initialize_matrix(n as usize);            
            // Call helper function
            exec_matrixmultiplication(&h_a, &h_b, n, &exec_config, true, &dev).unwrap();
            // Call helper function
            exec_matrixmultiplication(&h_a, &h_b, n, &exec_config, false, &dev).unwrap();
        }
    }
}