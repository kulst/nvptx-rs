use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use std::env;

fn main() -> Result<(), DriverError> {
    let mut args = env::args_os().skip(1);

    let dev = CudaDevice::new(0)?;

    // Load the kernel file specified in the first command line argument
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["add", "memcpy", "rgba2gray"],
    )?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("kernels", "rgba2gray").unwrap();

    let img = image::open(args.next().unwrap()).unwrap().to_rgba8();

    let width = img.width();
    let height = img.height();

    let mut img_grayscale = image::GrayImage::new(width, height);

    // Allocate device memory and copy host values to it
    let d_rgba = dev.htod_sync_copy(&img)?;
    let mut d_grayscale =
        dev.alloc_zeros::<u8>(usize::try_from(width).unwrap() * usize::try_from(height).unwrap())?;

    // Specify number of threads and launch the kernel let n = SIZE as u32;
    let (x_threads, y_threads, z_threads) = (32, 32, 1);
    let (x_blocks, y_blocks, z_blocks) = (
        (width + x_threads - 1) / x_threads,
        (height + y_threads - 1) / y_threads,
        1,
    );
    let cfg = LaunchConfig {
        grid_dim: (x_blocks, y_blocks, z_blocks),
        block_dim: (x_threads, y_threads, z_threads),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(
            cfg,
            (&d_rgba, &mut d_grayscale, width as i32, height as i32),
        )
    }?;

    // Deallocate device memory and copy it back to host if necessary
    dev.dtoh_sync_copy_into(&d_grayscale, &mut img_grayscale)?;
    let _ = dev.sync_reclaim(d_rgba)?;
    let _ = dev.sync_reclaim(d_grayscale)?;
    // Verify correctness

    img_grayscale.save(args.next().unwrap()).unwrap();

    Ok(())
}
