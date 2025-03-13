use clap::{command, error::ErrorKind, CommandFactory, Parser, ValueEnum};
use cudarc::{
    driver::{
        sys::*, CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    },
    nvrtc::Ptx,
};
use host_code::{himeno_stencil, CudaEvent, Matrix};
use num::{traits::float::FloatCore, FromPrimitive};
use rayon::{iter::repeatn, prelude::*};
use std::{
    any::TypeId,
    fmt::{Debug, Display},
    io::{self, Write},
    iter::{once, repeat, successors},
    sync::Arc,
};

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

/// 19 pt stencil as calculated in himeno benchmark computed on the GPU
/// (developed with Rust using the nvptx64-nvidia-cuda target)
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Smallest input size
    #[arg(long, value_enum)]
    from: Option<ParamSize>,
    /// Largest input size
    #[arg(long, value_enum, default_value_t = ParamSize::Xs)]
    to: ParamSize,
    /// Value type
    #[arg(long = "type", value_enum, default_value_t = ValueType::Both)]
    value_type: ValueType,
    /// Benchmark runs for a repititions number of times per parameter set
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    reps: u32,
    /// Benchmark runs for block-sizes in x from [xdim-min, xdim-min * 2, ... , xdim-max]
    #[arg(long, requires("ydim_min"), requires("ydim_max"), requires("xdim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    xdim_min: Option<u32>,
    /// Benchmark runs for block-sizes in x from [xdim-min, xdim-min * 2, ... , xdim-max]
    #[arg(long, requires("ydim_min"), requires("ydim_max"), requires("xdim_min"), value_parser = clap::value_parser!(u32).range(2..=256))]
    xdim_max: Option<u32>,
    /// Benchmark runs for block-sizes in y from [ydim-min, ydim-min * 2, ... , ydim-max]
    #[arg(long, requires("xdim_min"), requires("xdim_max"), requires("ydim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    ydim_min: Option<u32>,
    /// Benchmark runs for block-sizes in y from [ydim-min, ydim-min * 2, ... , ydim-max]
    #[arg(long, requires("ydim_min"), requires("xdim_min"), requires("xdim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    ydim_max: Option<u32>,
    /// Check the result against the one calculated on the host
    #[arg(long, default_value_t = false)]
    check: bool,
    /// Benchmark runs for grid-sizes in x from [grid-xdim-min, grid-xdim-min * 2, ... , grid-xdim-max]
    #[arg(long, requires("grid_ydim_min"), requires("grid_ydim_max"), requires("grid_xdim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    grid_xdim_min: Option<u32>,
    /// Benchmark runs for grid-sizes in x from [grid-xdim-min, grid-xdim-min * 2, ... , grid-xdim-max]
    #[arg(long, requires("grid_ydim_min"), requires("grid_ydim_max"), requires("grid_xdim_min"), value_parser = clap::value_parser!(u32).range(2..=256))]
    grid_xdim_max: Option<u32>,
    /// Benchmark runs for grid-sizes in y from [grid-ydim-min, grid-ydim-min * 2, ... , grid-ydim-max]
    #[arg(long, requires("grid_xdim_min"), requires("grid_xdim_max"), requires("grid_ydim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    grid_ydim_min: Option<u32>,
    /// Benchmark runs for grid-sizes in y from [grid-ydim-min, grid-ydim-min * 2, ... , grid-ydim-max]
    #[arg(long, requires("grid_ydim_min"), requires("grid_xdim_min"), requires("grid_xdim_max"), value_parser = clap::value_parser!(u32).range(2..=256))]
    grid_ydim_max: Option<u32>,
    /// Disable cyclic allocation on the device (helpful if only the kernel duration
    /// is of interest)
    #[arg(long, default_value_t = false, conflicts_with("check"))]
    no_cyclic_alloc: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, EnumIter)]
enum ParamSize {
    /// Grid-size (deps x rows x cols) = XS (32x32x64)
    Xs,
    /// Grid-size (deps x rows x cols) = S (64x64x128)
    S,
    /// Grid-size (deps x rows x cols) = M (128x128x256)
    M,
    /// Grid-size (deps x rows x cols) = L (256x256x512)
    L,
    /// Grid-size (deps x rows x cols) = XL (512x512x1024)
    Xl,
    /// Grid-size (deps x rows x cols) = XXL (1024x1024x512)
    Xxl,
    /// Grid-size (deps x rows x cols) = XXL (2048x1024x512)
    XxlPlus,
    /// Grid-size (deps x rows x cols) = XXXL (2048x2048x512)
    Xxxl,
}

impl ParamSize {
    fn get_size(&self) -> (usize, usize, usize) {
        match self {
            &ParamSize::Xs => (32, 32, 64),
            &ParamSize::S => (64, 64, 128),
            &ParamSize::M => (128, 128, 256),
            &ParamSize::L => (256, 256, 512),
            &ParamSize::Xl => (512, 512, 1024),
            &ParamSize::Xxl => (1024, 1024, 512),
            &ParamSize::XxlPlus => (2048, 1024, 512),
            ParamSize::Xxxl => (2048, 2048, 512),
        }
    }
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
struct ExecConfig<'a> {
    block_dims: &'a [(u32, u32)],
    grid_xdim_min: Option<u32>,
    grid_xdim_max: Option<u32>,
    grid_ydim_min: Option<u32>,
    grid_ydim_max: Option<u32>,
    reps: u32,
    no_cyclic_alloc: bool,
}

struct HimenoInputs<T> {
    a0: T,
    a1: T,
    a2: T,
    a3: T,
    b: T,
    c: T,
    bnd: T,
    wrk1: T,
    omega: T,
}
//
// Helper function to execute the kernel depending on type, configuration etc.
fn exec_stencil<T>(
    input: &Matrix<T>,
    exec_cfg: &ExecConfig,
    himeno_inputs: &HimenoInputs<T>,
    result: Option<&Matrix<T>>,
    device: &Arc<CudaDevice>,
) -> Result<(), DriverError>
where
    T: FloatCore
        + 'static
        + DeviceRepr
        + ValidAsZeroBits
        + Default
        + Unpin
        + Debug
        + Display
        + Send
        + Sync
        + std::iter::Sum
        + FromPrimitive,
{
    // We lock the stdout only one time per function invocation to improve performance
    let mut stdout = io::stdout().lock();
    // We must choose the right kernel depending on type of T
    let (kernel, kernel_name) = if TypeId::of::<T>() == TypeId::of::<f32>() {
        (
            device.get_func("kernels", "stencil_f32").unwrap(),
            "stencil_f32",
        )
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        (
            device.get_func("kernels", "stencil_f64").unwrap(),
            "stencil_f64",
        )
    } else {
        unreachable!()
    };
    // get input size
    let cols = input.get_cols();
    let rows = input.get_rows();
    let deps = input.get_deps();
    // Create events to measure times
    let mut pre_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut pre_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_kernel_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    let mut post_event = CudaEvent::new(device.clone(), CUevent_flags::CU_EVENT_DEFAULT)?;
    // if cyclic allocation is disabled we allocate the device buffers here already
    let mut device_buffers = if exec_cfg.no_cyclic_alloc {
        Some((
            device.htod_sync_copy(input.as_slice())?,
            device.alloc_zeros::<T>(input.size())?,
        ))
    } else {
        None
    };
    // destructure the himeno stencil inputs
    let HimenoInputs {
        a0,
        a1,
        a2,
        a3,
        b,
        c,
        bnd,
        wrk1,
        omega,
    } = *himeno_inputs;
    // Iterate over the given block sizes
    for &(block_dim_x, block_dim_y) in exec_cfg.block_dims {
        // We calculate the grid dimensions we iterate over for the given block dimension
        let grid_dims : Vec<(u32, u32)> = if let (Some(grid_xdim_min),Some(grid_xdim_max),Some(grid_ydim_min), Some(grid_ydim_max)) = 
            (exec_cfg.grid_xdim_min, exec_cfg.grid_xdim_max, exec_cfg.grid_ydim_min, exec_cfg.grid_ydim_max) {
        let grid_dims_x = successors(Some(grid_xdim_min), |&dim| Some(dim << 1))
            .take_while(|&dim| dim < grid_xdim_max)
            .chain(once(grid_xdim_max));
        let grid_dims_y = successors(Some(grid_ydim_min), |&dim| Some(dim << 1))
            .take_while(|&dim| dim < grid_ydim_max)
            .chain(once(grid_ydim_max));
        grid_dims_x
            .map(|dim| repeat(dim).zip(grid_dims_y.clone()))
            .flatten()
            .filter(|&(dim_x, dim_y)| dim_x * dim_y <= 2048)
            .collect()
        } else {
            let grid_dim_x = (cols as u32 + block_dim_x - 1) / block_dim_x;
            let grid_dim_y = (rows as u32 + block_dim_y - 1) / block_dim_y;
            once((grid_dim_x,grid_dim_y)).collect()
        };

        // Iterate over grid dimensions
        for (grid_dim_x, grid_dim_y) in grid_dims {
            // Create the launch configuration, shared memory must be large enough to hold
            // one element per thread
            let cfg = LaunchConfig {
                grid_dim: (grid_dim_x, grid_dim_y, 1),
                block_dim: (block_dim_x, block_dim_y, 1),
                shared_mem_bytes: ((block_dim_x + 2) as usize
                    * (block_dim_y + 2) as usize
                    * 3
                    * size_of::<T>()) as u32,
            };
            // Iterate over the given number of repititions
            for i in 0..exec_cfg.reps {
                // If device input and device output are already allocated, we skip cyclic allocation
                let kernel_result = if let Some((d_input, d_output)) = &mut device_buffers {
                    // Record event before calling into CUDA
                    pre_event.record()?;
                    // Record event before kernel invocation
                    pre_kernel_event.record()?;
                    unsafe {
                        kernel.clone().launch(
                            cfg,
                            (
                                &*d_input,
                                a0,
                                a1,
                                a2,
                                a3,
                                b,
                                c,
                                wrk1,
                                bnd,
                                d_output,
                                omega,
                                deps as i32,
                                rows as i32,
                                cols as i32,
                            ),
                        )?
                    }
                    // Record event after kernel invocation
                    post_kernel_event.record()?;
                    // Record event after calling into CUDA and sync the device
                    post_event.record()?;
                    post_event.sync()?;
                    None
                } else {
                    // Record event before calling into CUDA
                    pre_event.record()?;
                    let d_input = device.htod_sync_copy(input.as_slice())?;
                    let mut d_output = device.alloc_zeros::<T>(input.size())?;
                    // Record event before kernel invocation
                    pre_kernel_event.record()?;
                    unsafe {
                        kernel.clone().launch(
                            cfg,
                            (
                                &d_input,
                                a0,
                                a1,
                                a2,
                                a3,
                                b,
                                c,
                                wrk1,
                                bnd,
                                &mut d_output,
                                omega,
                                deps as i32,
                                rows as i32,
                                cols as i32,
                            ),
                        )?
                    }
                    // Record event after kernel invocation
                    post_kernel_event.record()?;
                    // Copy back result and drop the device input buffer
                    let kernel_result = device.sync_reclaim(d_output)?;
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
                let mut differences: Vec<(usize, T)> = Vec::new();
                let output = match (result, kernel_result) {
                    (Some(result), Some(kernel_result)) => {
                    kernel_result
                        .par_iter()
                        .zip(result.as_slice().par_iter())
                        .enumerate()
                        .map(|(index, (&dev_res, &host_res))| {
                            let dif = (dev_res - host_res).abs();
                            (index, dif)
                        })
                        .collect_into_vec(&mut differences);
                    let difference: T = differences.par_iter().map(|(_, dif)| dif).cloned().sum();
                    format!(
                        "{kernel_name};{i};{cols};{rows};{deps};{block_dim_x};{block_dim_y};{grid_dim_x};{grid_dim_y};{pre_dur};{kernel_dur};{post_dur};{difference}\n"
                    )                    
                    }
                    (None, None) => format!(
                        "{kernel_name};{i};{cols};{rows};{deps};{block_dim_x};{block_dim_y};{grid_dim_x};{grid_dim_y};{kernel_dur}\n"
                    ),
                    (None, Some(_)) => 
                    format!(
                        "{kernel_name};{i};{cols};{rows};{deps};{block_dim_x};{block_dim_y};{grid_dim_x};{grid_dim_y};{pre_dur};{kernel_dur};{post_dur}\n"
                    ),
                    (Some(_), None) => unreachable!(), 
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
        value_type,
        reps,
        xdim_min,
        xdim_max,
        ydim_min,
        ydim_max,
        check,
        no_cyclic_alloc,
        grid_xdim_min,
        grid_xdim_max,
        grid_ydim_min,
        grid_ydim_max,
    } = Cli::parse();
    let from = from.unwrap_or(to);
    // Validate the inputs
    if to < from {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "to must be greater or equal from",
            )
            .exit();
    }

    if (to >= ParamSize::Xxxl || from >= ParamSize::Xxxl) && (value_type == ValueType::Both || value_type == ValueType::F64) {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "xxxl is not supported with f64",
            )
            .exit();
    }
    
    let (xdim_min, xdim_max, ydim_min, ydim_max) = match (xdim_min, xdim_max, ydim_min, ydim_max) {
        (Some(xdim_min), Some(xdim_max), Some(ydim_min), Some(ydim_max)) => {
            (xdim_min, xdim_max, ydim_min, ydim_max)
        },
        (None, None, None, None) => (128, 128, 4, 4),
        _ => unreachable!(),
    };
    if xdim_min > xdim_max || ydim_min > ydim_max {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "dim_max must be greater or equal dim_min",
            )
            .exit();
    }
    if grid_xdim_min > grid_xdim_max || grid_ydim_min > grid_ydim_max {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "grid_dim_max must be greater or equal grid_dim_min",
            )
            .exit();
    }
    if !(xdim_min * ydim_min).is_power_of_two()
        || !(xdim_max * ydim_max).is_power_of_two()
        || (xdim_min * ydim_min) > 512
    {
        Cli::command()
            .error(
                ErrorKind::ArgumentConflict,
                "xdim * ydim must be a power of 2 and smaller or equal 512",
            )
            .exit();
    }
    if let (Some(grid_xdim_min),Some(grid_xdim_max),Some(grid_ydim_min), Some(grid_ydim_max)) = 
        (grid_xdim_min, grid_xdim_max, grid_ydim_min, grid_ydim_max) {
        if !(grid_xdim_min * grid_ydim_min).is_power_of_two()
            || !(grid_xdim_max * grid_ydim_max).is_power_of_two()
            || (grid_xdim_min * grid_ydim_min) > 2048
        {
            Cli::command()
                .error(
                    ErrorKind::ArgumentConflict,
                    "grid_xdim * grid_ydim must be a power of 2 and smaller or equal 2048",
                )
                .exit();
        }
    }

    // Calculate the input sizes to iterate on
    let input_sizes = ParamSize::iter()
        .filter_map(|size| {
            if size <= to && size >= from {
                return Some(size.get_size());
            } else {
                return None;
            }
        });
    let block_dims_x: Vec<u32> = successors(Some(xdim_min), |&dim| Some(dim << 1))
        .take_while(|&dim| dim <= xdim_max)
        .collect();
    let block_dims_y: Vec<u32> = successors(Some(ydim_min), |&dim| Some(dim << 1))
        .take_while(|&dim| dim <= ydim_max)
        .collect();
    let block_dims: Vec<(u32, u32)> = block_dims_x
        .iter()
        .map(|&x_dim| {
            let dims: Vec<(u32, u32)> = repeat(x_dim)
                .zip(block_dims_y.iter())
                .map(|(x_dim, &y_dim)| (x_dim, y_dim))
                .filter(|(x_dim, y_dim)| (x_dim * y_dim <= 512) && (x_dim * y_dim) >= 32)
                .collect();
            dims
        })
        .flatten()
        .collect();
    // Create the execution configuration
    let exec_config = ExecConfig {
        block_dims: &block_dims,
        reps,
        no_cyclic_alloc,
        grid_xdim_min,
        grid_xdim_max,
        grid_ydim_min,
        grid_ydim_max,
    };
    // Create the Cuda Device
    let dev = CudaDevice::new(0).unwrap();
    // Load the kernel file and compile it
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["stencil_f32", "stencil_f64"],
    )
    .unwrap();

    // Print the header
    match (check, no_cyclic_alloc) {
        (true, true) => unreachable!(),
        (true, false) => println!("kernel;iteration;cols;rows;deps;block-dim-x;block-dim-y;grid-dim-x;grid-dim-y;pre-duration;kernel-duration;post-duration;result-difference"),
        (false, true) => println!("kernel;iteration;cols;rows;deps;block-dim-x;block-dim-y;grid-dim-x;grid-dim-y;kernel-duration"),
        (false, false) => println!("kernel;iteration;cols;rows;deps;block-dim-x;block-dim-y;grid-dim-x;grid-dim-y;pre-duration;kernel-duration;post-duration"),
    }
    // Iterate over input sizes
    for (deps, rows, cols) in input_sizes {
        // Call helper function for f32 type if necessary
        if value_type == ValueType::F32 || value_type == ValueType::Both {
            // Generate input vec on host 
            let input = (0..deps)
                .into_par_iter()
                .flat_map(|deps_idx| repeatn(deps_idx, cols * rows))
                .map(|deps_idx| (deps_idx * deps_idx) as f32 / ((deps - 1) * (deps - 1)) as f32)
                .collect();
            let input = Matrix::from_vec(input, cols, rows, deps).unwrap();
            // Set stencil input parameters
            let (a0, a1, a2, a3, b, c, bnd, wrk1, omega) = (
                1f32,
                1f32,
                1f32,
                1f32 / 6f32,
                0f32,
                1f32,
                1f32,
                0f32,
                0.8f32,
            );
            // If we need to check the result, calculate the stencil on the host
            let result = if check {
                Some(himeno_stencil(
                    &input, a0, a1, a2, a3, b, c, wrk1, bnd, omega,
                ))
            } else {
                None
            };
            // Set HimenoInputs struct and call helper function 
            let himeno_inputs = HimenoInputs {
                a0,
                a1,
                a2,
                a3,
                b,
                c,
                bnd,
                wrk1,
                omega,
            };
            exec_stencil(&input, &exec_config, &himeno_inputs, result.as_ref(), &dev).unwrap();
        }
        // Call helper function for f64 type if necessary
        if value_type == ValueType::F64 || value_type == ValueType::Both {
            // Generate input vec on host 
            let input = (0..deps)
                .into_par_iter()
                .flat_map(|deps_idx| repeatn(deps_idx, cols * rows))
                .map(|deps_idx| (deps_idx * deps_idx) as f64 / ((deps - 1) * (deps - 1)) as f64)
                .collect();
            let input = Matrix::from_vec(input, cols, rows, deps).unwrap();
            // Set stencil input parameters
            let (a0, a1, a2, a3, b, c, bnd, wrk1, omega) = (
                1f64,
                1f64,
                1f64,
                1f64 / 6f64,
                0f64,
                1f64,
                1f64,
                0f64,
                0.8f64,
            );
            // If we need to check the result, calculate the stencil on the host
            let result = if check {
                Some(himeno_stencil(
                    &input, a0, a1, a2, a3, b, c, wrk1, bnd, omega,
                ))
            } else {
                None
            };
            // Set HimenoInputs struct and call helper function 
            let himeno_inputs = HimenoInputs {
                a0,
                a1,
                a2,
                a3,
                b,
                c,
                bnd,
                wrk1,
                omega,
            };
            exec_stencil(&input, &exec_config, &himeno_inputs, result.as_ref(), &dev).unwrap();
        }
    }
}
