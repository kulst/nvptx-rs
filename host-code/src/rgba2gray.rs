use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::{
    env,
    ops::{Deref, DerefMut},
};

fn main() {
    let mut args = env::args_os().skip(1);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )))
        .unwrap();

    // and then retrieve the function with `get_func`
    let f = module.load_function("rgba2gray").unwrap();

    let img = image::open(args.next().unwrap()).unwrap().to_rgba8();

    let width = img.width();
    let height = img.height();

    let mut img_grayscale = image::GrayImage::new(width, height);

    // Allocate device memory and copy host values to it
    let d_rgba = stream.memcpy_stod(img.deref()).unwrap();
    let mut d_grayscale = stream
        .alloc_zeros::<u8>(usize::try_from(width).unwrap() * usize::try_from(height).unwrap())
        .unwrap();

    // Specify number of threads and launch the kernel let n = SIZE as u32;
    let (x_threads, y_threads, z_threads) = (32, 32, 1);
    let (x_blocks, y_blocks, z_blocks) = (width.div_ceil(x_threads), height.div_ceil(y_threads), 1);
    let cfg = LaunchConfig {
        grid_dim: (x_blocks, y_blocks, z_blocks),
        block_dim: (x_threads, y_threads, z_threads),
        shared_mem_bytes: 0,
    };
    let mut launch_args = stream.launch_builder(&f);
    let (height, width) = (height as i32, width as i32);
    launch_args
        .arg(&d_rgba)
        .arg(&mut d_grayscale)
        .arg(&width)
        .arg(&height);
    unsafe { launch_args.launch(cfg).unwrap() };

    // Deallocate device memory and copy it back to host if necessary
    stream
        .memcpy_dtoh(&d_grayscale, img_grayscale.deref_mut())
        .unwrap();
    drop(d_grayscale);
    drop(d_rgba);
    // Verify correctness

    img_grayscale.save(args.next().unwrap()).unwrap();
}
