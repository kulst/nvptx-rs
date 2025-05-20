use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use rand::{distributions::Standard, prelude::*};

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )))
        .unwrap();

    // and then retrieve the function with `get_func`
    let f = module.load_function("memcpy").unwrap();

    // Specify size of input array
    const SIZE: usize = 1024 * 1024;

    // Allocate host memory and populate it (input arrays with random data, output array with 0)
    let mut rng = rand::thread_rng().sample_iter(Standard);

    let h_a: Vec<f32> = rng.by_ref().take(SIZE).collect();
    let mut h_b: Vec<f32> = std::iter::repeat_n(0f32, SIZE).collect();

    // Allocate device memory and copy host values to it
    let d_a = stream.memcpy_stod(&h_a).unwrap();
    let mut d_b = stream.alloc_zeros::<f32>(SIZE).unwrap();

    // Specify number of threads and launch the kernel
    let n = SIZE;
    let cfg = LaunchConfig::for_num_elems(n.try_into().unwrap());
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut d_b).arg(&d_a).arg(&n);
    unsafe { launch_args.launch(cfg).unwrap() };

    // Deallocate device memory and copy it back to host if necessary
    stream.memcpy_dtoh(&d_b, &mut h_b).unwrap();
    drop(d_a);
    drop(d_b);
    // Verify correctness
    assert_eq!(h_a, h_b);
}
