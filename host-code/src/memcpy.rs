use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use rand::{distributions::Standard, prelude::*};
use std::iter;

fn main() -> Result<(), DriverError> {
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
    let f = dev.get_func("kernels", "memcpy").unwrap();

    // Specify size of input array
    const SIZE: usize = 1024 * 1024;

    // Allocate host memory and populate it (input arrays with random data, output array with 0)
    let mut rng = rand::thread_rng().sample_iter(Standard);

    let h_a: Vec<f32> = rng.by_ref().take(SIZE).collect();
    let mut h_b: Vec<f32> = iter::repeat(0f32).take(SIZE).collect();

    // Allocate device memory and copy host values to it
    let d_a = dev.htod_sync_copy(&h_a)?;
    let mut d_b = dev.alloc_zeros::<f32>(SIZE)?;

    // Specify number of threads and launch the kernel
    let n = SIZE as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut d_b, &d_a, n as usize)) }?;

    // Deallocate device memory and copy it back to host if necessary
    dev.sync_reclaim(d_a)?;
    dev.dtoh_sync_copy_into(&d_b, &mut h_b)?;

    // Verify correctness
    assert_eq!(h_a, h_b);
    Ok(())
}
