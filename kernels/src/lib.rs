#![feature(abi_ptx, stdarch_nvptx)]
#![no_std]

use core::arch::nvptx::*;

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
