use cudarc::{
    driver::{
        result::texture::create_object, sys::*, CudaDevice, DriverError, LaunchAsync, LaunchConfig,
    },
    nvrtc::Ptx,
};
use rand::{distributions::Standard, prelude::*};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    // Load the kernel file specified in the first command line argument
    dev.load_ptx(
        Ptx::from_src(include_str!(
            "../../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
        )),
        "kernels",
        &["add", "memcpy", "rgba2gray", "texture_memcpy"],
    )?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("kernels", "texture_memcpy").unwrap();

    // Specify size of input array
    const SIZE: usize = 1024 * 1024;

    // Allocate host memory and populate it (input arrays with random data, output array with 0)
    let mut rng = rand::thread_rng().sample_iter(Standard);

    let h_a: Vec<f32> = rng.by_ref().take(SIZE).collect();

    // Allocate device memory and copy host values to it
    let mut d_b = dev.alloc_zeros::<f32>(SIZE)?;
    let d_a_p;
    unsafe {
        d_a_p = cudarc::driver::result::malloc_sync(std::mem::size_of::<f32>() * SIZE).unwrap();
        cudarc::driver::result::memcpy_htod_sync(d_a_p, &h_a).unwrap();
    }
    let resource_desc = CUDA_RESOURCE_DESC {
        resType: CUresourcetype::CU_RESOURCE_TYPE_LINEAR,
        res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
            linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
                devPtr: d_a_p,
                format: CUarray_format_enum::CU_AD_FORMAT_FLOAT,
                numChannels: 1,
                sizeInBytes: std::mem::size_of::<f32>() * SIZE,
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
    let n = 4096;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { f.launch(cfg, (&mut d_b, texture_object, SIZE as usize)) }?;

    // Deallocate device memory and copy it back to host if necessary
    let h_b = dev.sync_reclaim(d_b)?;

    unsafe {
        cudarc::driver::result::texture::destroy_object(texture_object).unwrap();
        cudarc::driver::result::free_sync(d_a_p).unwrap();
    }

    // Verify correctness
    assert_eq!(h_a, h_b);

    Ok(())
}
