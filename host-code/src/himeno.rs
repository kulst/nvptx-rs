use clap::{command, Parser, ValueEnum};
use cudarc::{
    driver::{
        result::texture::create_object, sys::*, CudaDevice, DriverError, LaunchAsync, LaunchConfig,
    },
    nvrtc::Ptx,
};
use rayon::{iter::repeatn, prelude::*};
use std::ops::Index;
// Himeno benchmark in Rust using nvptx
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// The size of the benchmark
    #[arg(value_enum)]
    size: ParamSize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ParamSize {
    /// Grid-size = XS (32x32x64)
    XS,
    /// Grid-size = S (64x64x128)
    S,
    /// Grid-size = M (128x128x256)
    M,
    /// Grid-size = L (256x256x512)
    L,
    /// Grid-size = XL (512x512x1024)
    XL,
}

struct Matrix {
    buffer: Vec<f32>,
    nums: usize,
    rows: usize,
    cols: usize,
    deps: usize,
}

impl Matrix {
    fn new(nums: usize, rows: usize, cols: usize, deps: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(nums * rows * cols * deps),
            nums,
            rows,
            cols,
            deps,
        }
    }

    fn size(&self) -> usize {
        self.nums * self.rows * self.cols * self.deps
    }
}

impl Index<(usize, usize, usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        &self.buffer[index.0 * self.rows * self.cols * self.deps
            + index.1 * self.cols * self.deps
            + index.2 * self.deps
            + index.3]
    }
}

fn main() -> Result<(), DriverError> {
    let args = Cli::parse();

    let (rows, cols, deps) = match args.size {
        ParamSize::XS => (32usize, 32usize, 64usize),
        ParamSize::S => (64usize, 64usize, 128usize),
        ParamSize::M => (128usize, 128usize, 256usize),
        ParamSize::L => (256usize, 256usize, 512usize),
        ParamSize::XL => (512usize, 512usize, 1024usize),
    };

    let mut p: Matrix = Matrix::new(1, rows, cols, deps);
    let mut bnd: Matrix = Matrix::new(1, rows, cols, deps);
    let mut wrk1: Matrix = Matrix::new(1, rows, cols, deps);
    let mut a: Matrix = Matrix::new(4, rows, cols, deps);
    let mut b: Matrix = Matrix::new(3, rows, cols, deps);
    let mut c: Matrix = Matrix::new(3, rows, cols, deps);

    p.buffer = (0..p.rows)
        .into_par_iter()
        .flat_map(|row_idx| repeatn(row_idx, p.cols * p.deps))
        .map(|row_idx| (row_idx * row_idx) as f32 / ((p.rows - 1) * (p.rows - 1)) as f32)
        .collect();
    println!("{}", p.buffer.len());
    repeatn(1.0f32, bnd.size()).collect_into_vec(&mut bnd.buffer);
    println!("{}", bnd.buffer.len());
    repeatn(0f32, wrk1.size()).collect_into_vec(&mut wrk1.buffer);
    println!("{}", wrk1.buffer.len());

    a.buffer = (0..a.nums)
        .into_par_iter()
        .flat_map(|num_idx| repeatn(num_idx, a.rows * a.cols * a.deps))
        .map(|num_idx| if num_idx < 3 { 1.0f32 } else { 1.0f32 / 6.0f32 })
        .collect();
    println!("{}", a.buffer.len());
    repeatn(0.0f32, b.size()).collect_into_vec(&mut b.buffer);
    println!("{}", b.buffer.len());
    repeatn(1.0f32, c.size()).collect_into_vec(&mut c.buffer);
    println!("{}", c.buffer.len());

    let dev = CudaDevice::new(0)?;

    // Load the kernel file specified in the first command line argument
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["himeno"],
    )?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("kernels", "himeno").unwrap();

    // Allocate device memory and copy host values to it
    let d_p;
    unsafe {
        d_p = cudarc::driver::result::malloc_sync(
            std::mem::size_of::<f32>() * p.rows * p.cols * p.deps,
        )
        .unwrap();
        cudarc::driver::result::memcpy_htod_sync(d_p, &p.buffer).unwrap();
    }
    let resource_desc = CUDA_RESOURCE_DESC {
        resType: CUresourcetype::CU_RESOURCE_TYPE_LINEAR,
        res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
            linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
                devPtr: d_p,
                format: CUarray_format_enum::CU_AD_FORMAT_FLOAT,
                numChannels: 1,
                sizeInBytes: std::mem::size_of::<f32>() * p.rows * p.cols * p.deps,
            },
        },
        flags: 0,
    };

    let texture_desc = CUDA_TEXTURE_DESC {
        addressMode: [
            CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
            CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
            CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
        ],
        filterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
        flags: 0,
        maxAnisotropy: 0,
        mipmapFilterMode: CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
        mipmapLevelBias: 0f32,
        minMipmapLevelClamp: 0f32,
        maxMipmapLevelClamp: 0f32,
        borderColor: [0f32, 0f32, 0f32, 0f32],
        reserved: [0; 12],
    };
    let texture_object =
        unsafe { create_object(&resource_desc, &texture_desc, std::ptr::null()).unwrap() };
    // Specify number of threads and launch the kernel

    let d_a = dev.htod_sync_copy(&a.buffer).unwrap();
    let d_b = dev.htod_sync_copy(&b.buffer).unwrap();
    let d_c = dev.htod_sync_copy(&c.buffer).unwrap();
    let d_bnd = dev.htod_sync_copy(&bnd.buffer).unwrap();
    let d_wrk1 = dev.htod_sync_copy(&wrk1.buffer).unwrap();
    let mut d_wrk2 = dev.alloc_zeros::<f32>(rows * cols * deps).unwrap();
    let mut d_gosa = dev.alloc_zeros::<f32>(1).unwrap();
    let omega = 0.8f32;
    let (threads_x, threads_y) = (128, 4);
    let cfg = LaunchConfig {
        grid_dim: (
            (deps as u32 + threads_x - 1) / threads_x,
            (cols as u32 + threads_y - 1) / threads_y,
            1,
        ),
        block_dim: (threads_x, threads_y, 1),
        shared_mem_bytes: std::mem::size_of::<f32>() as u32 * (threads_x + 2) * (threads_y + 2) * 3,
    };

    unsafe {
        f.launch(
            cfg,
            (
                texture_object,
                &d_a,
                &d_b,
                &d_c,
                &d_wrk1,
                &d_bnd,
                &mut d_wrk2,
                &mut d_gosa,
                omega,
                rows as i32, // i
                cols as i32, // j
                deps as i32, // k
            ),
        )
    }
    .unwrap();

    // Deallocate device memory and copy it back to host if necessary
    dev.sync_reclaim(d_a).unwrap();
    dev.sync_reclaim(d_b).unwrap();
    dev.sync_reclaim(d_c).unwrap();
    dev.sync_reclaim(d_bnd).unwrap();
    dev.sync_reclaim(d_wrk1).unwrap();
    dev.sync_reclaim(d_wrk2).unwrap();
    let gosa = dev.sync_reclaim(d_gosa).unwrap()[0];

    unsafe {
        cudarc::driver::result::texture::destroy_object(texture_object).unwrap();
        cudarc::driver::result::free_sync(d_p).unwrap();
    }

    // Verify correctness
    println!("gosa: {gosa}");

    Ok(())
}
