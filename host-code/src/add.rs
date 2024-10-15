use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use rand::{distributions::Standard, prelude::*};
use std::env;

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // Load the kernel file specified in the first command line argument
    dev.load_ptx(
        Ptx::from_file(env::args_os().skip(1).next().unwrap()),
        "kernels",
        &["add", "memcpy", "rgba2gray"],
    )?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("kernels", "add").unwrap();

    // Specify size of input array
    const SIZE: usize = 1024 * 1024;

    // Allocate host memory and populate it (input arrays with random data, output array with 0)
    let mut rng = rand::thread_rng().sample_iter(Standard);

    let h_a: Vec<f32> = rng.by_ref().take(SIZE).collect();
    let h_b: Vec<f32> = rng.by_ref().take(SIZE).collect();
    let mut h_c: Vec<f32> = (0..SIZE).map(|_| 0.).collect();

    // Allocate device memory and copy host values to it
    let d_a = dev.htod_sync_copy(&h_a)?;
    let d_b = dev.htod_sync_copy(&h_b)?;
    let mut d_c = dev.htod_sync_copy(&h_c)?;

    // Specify number of threads and launch the kernel
    let n = SIZE as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&d_a, &d_b, &mut d_c, n as usize)) }?;

    // Deallocate device memory and copy it back to host if necessary
    dev.sync_reclaim(d_a)?;
    dev.sync_reclaim(d_b)?;
    dev.dtoh_sync_copy_into(&d_c, &mut h_c)?;
    dev.sync_reclaim(d_c)?;

    // Perform the same computation on the host
    let c = h_a.iter().zip(h_b).map(|(a, b)| a + b).collect::<Vec<_>>();

    // Verify correctness
    assert_eq!(c, h_c);
    Ok(())
}
