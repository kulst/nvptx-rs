#![feature(abi_ptx, stdarch_nvptx)]
#![feature(asm_experimental_arch)]
#![feature(const_type_id)]
#![feature(core_intrinsics)]
#![feature(shared_memory)]
#![no_std]

use core::arch::nvptx::*;

mod intrinsics;
mod linear;
mod matrixmultiplication;
mod reduction;
mod stencil;
mod util;

/// Kernel to calculate the 19 pt stencil of the himeno benchmark with f32 values
#[no_mangle]
pub unsafe extern "ptx-kernel" fn stencil_f32(
    p: *const f32,
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    b: f32,
    c: f32,
    wrk1: f32,
    bnd: f32,
    wrk2: *mut f32,
    omega: f32,
    i: i32,
    j: i32,
    k: i32,
) {
    stencil::stencil(p, a0, a1, a2, a3, b, c, wrk1, bnd, wrk2, omega, i, j, k);
}
/// Kernel to calculate the 19 pt stencil of the himeno benchmark with f64 values
#[no_mangle]
pub unsafe extern "ptx-kernel" fn stencil_f64(
    p: *const f64,
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    b: f64,
    c: f64,
    wrk1: f64,
    bnd: f64,
    wrk2: *mut f64,
    omega: f64,
    i: i32,
    j: i32,
    k: i32,
) {
    stencil::stencil(p, a0, a1, a2, a3, b, c, wrk1, bnd, wrk2, omega, i, j, k);
}

/// Kernel to calculate a simple reduction with f32 values
#[no_mangle]
pub unsafe extern "ptx-kernel" fn reduction_f32(input: *const f32, output: *mut f32, n: usize) {
    reduction::reduction(input, output, n);
}
/// Kernel to calculate a simple reduction with f64 values
#[no_mangle]
pub unsafe extern "ptx-kernel" fn reduction_f64(input: *const f64, output: *mut f64, n: usize) {
    reduction::reduction(input, output, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrixmultiplication_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: usize,
) {
    matrixmultiplication::matrixmultiplication(a, b, c, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrixmultiplication_shared_memory_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: usize,
) {
    matrixmultiplication::matrixmultiplication_shared_memory(a, b, c, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrixmultiplication_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    n: usize,
) {
    matrixmultiplication::matrixmultiplication(a, b, c, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn matrixmultiplication_shared_memory_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    n: usize,
) {
    matrixmultiplication::matrixmultiplication_shared_memory(a, b, c, n);
}

/// Add two "vectors" of length `n`. `c <- a + b`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let i = _block_dim_x()
        .wrapping_mul(_block_idx_x())
        .wrapping_add(_thread_idx_x());

    if (i as usize) >= n {
        return;
    }
    *c.offset(i as isize) = *a.offset(i as isize) + *b.offset(i as isize);
}

/// Copies an array of `n` floating point numbers from `src` to `dst`
#[no_mangle]
pub unsafe extern "ptx-kernel" fn memcpy(dst: *mut f32, src: *const f32, n: usize) {
    let i = _block_dim_x()
        .wrapping_mul(_block_idx_x())
        .wrapping_add(_thread_idx_x());

    if (i as usize) >= n {
        return;
    }
    *dst.offset(i as isize) = *src.offset(i as isize);
}

#[repr(C)]
pub struct Rgba {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn rgba2gray(
    rgba: *const Rgba,
    gray: *mut u8,
    width: i32,
    height: i32,
) {
    let x = _block_dim_x()
        .wrapping_mul(_block_idx_x())
        .wrapping_add(_thread_idx_x());
    let y = _block_dim_y()
        .wrapping_mul(_block_idx_y())
        .wrapping_add(_thread_idx_y());

    if x >= width || y >= height {
        return;
    }

    let i = y.wrapping_mul(width).wrapping_add(x) as isize;

    let Rgba { r, g, b, .. } = *rgba.offset(i);

    *gray.offset(i) = (0.299 * f32::from(r) + 0.589 * f32::from(g) + 0.114 * f32::from(b)) as u8;
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
