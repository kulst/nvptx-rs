#![feature(abi_ptx, stdarch_nvptx)]
#![feature(asm_experimental_arch)]
#![no_std]

use core::{arch::nvptx::*, ffi::c_ulonglong};

use intrinsics::{_atomic_add_f32, _dynamic_shared_mem, _init_dyn_shared_mem};
use linear::*;
use tex_object::*;
use util::*;

mod intrinsics;
mod linear;
mod tex_object;
mod util;

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

#[no_mangle]
pub unsafe extern "ptx-kernel" fn texture_memcpy(dst: *mut f32, src: c_ulonglong, n: usize) {
    let thread_count = _block_dim_x() * _grid_dim_x();
    let mut id = _block_dim_x()
        .wrapping_mul(_block_idx_x())
        .wrapping_add(_thread_idx_x());
    while (id as usize) < n {
        *dst.offset(id as isize) = intrinsics::_tex_1d_fetch_f32(src, id as i32);
        id = id.wrapping_add(thread_count);
    }
}

pub type TexObject = u64;

/// kernel to calculate one iteration of the himeno benchmark
///
/// SAFETY:
///
/// - dynamic shared memory must be of the size
///     `size_of::<f32>() * (_block_dim_x() + 2) * (_block_dim_y() + 2) * 3`
/// - kernel must be called with _block_dim_x() > 1 and _block_dim_y() > 1
/// - kernel should not be called with _block_dim_z() > 1
#[no_mangle]
pub unsafe extern "ptx-kernel" fn himeno(
    p: TexObject,
    a: *const f32,
    b: *const f32,
    c: *const f32,
    wrk1: *const f32,
    bnd: *const f32,
    wrk2: *mut f32,
    gosa: *mut f32,
    omega: f32,
    i: i32,
    j: i32,
    k: i32,
) {
    // initialize dynamic shared memory
    _init_dyn_shared_mem();
    // thread id x in block
    let tid_x = _thread_idx_x() as isize;
    // thread id y in block
    let tid_y = _thread_idx_y() as isize;
    // thread id in block
    let tid = tid_y * _block_dim_x() as isize + tid_x;
    // number of threads per block
    let nthreads = (_block_dim_x() * _block_dim_y()) as isize;
    // thread id x in grid
    let gtid_x = _block_dim_x() * _block_idx_x() + _thread_idx_x();
    // number of threads in x in grid
    let gnthreads_x = _block_dim_x() * _grid_dim_x();
    // thread id y in grid
    let gtid_y = _block_dim_y() * _block_idx_y() + _thread_idx_y();
    // number of threads in y in grid
    let gnthreads_y = _block_dim_y() * _grid_dim_y();
    // number of blocks in x necessary to process the input
    let nblocks_x = (k - 2 + _block_dim_x() - 1) / _block_dim_x();
    // number of blocks in y necessary to process the input
    let nblocks_y = (j - 2 + _block_dim_y() - 1) / _block_dim_y();
    // residual
    let mut residual = 0f32;
    // associate p as a TexObject with 3 dimensions
    let mut p = TexObjectF32_3D::new(p, k, j, i);
    // shared memory rows for p_sh_[top,mid,bot]
    let smem_rows = (_block_dim_y() + 2) as isize;
    // shared memory columns for p_sh_[top,mid,bot]
    let smem_cols = (_block_dim_x() + 2) as isize;
    // number of items for p_sh_[top,mid,bot]
    let smem_len = (smem_rows * smem_cols) as usize;
    // associate p_sh_top as Linear2D in shared memory
    let p_sh_top = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_top = Linear2D::new(p_sh_top, smem_cols, smem_rows);
    // associate p_sh_mid as Linear2D in shared memory
    let p_sh_mid = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_mid = Linear2D::new(p_sh_mid, smem_cols, smem_rows);
    // associate p_sh_bot as Linear2D in shared memory
    let p_sh_bot = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_bot = Linear2D::new(p_sh_bot, smem_cols, smem_rows);
    // associate other arrays as Linear4D or Linear3D
    let a = Linear4D::new(a, k as isize, j as isize, i as isize, 4);
    let b = Linear4D::new(b, k as isize, j as isize, i as isize, 3);
    let c = Linear4D::new(c, k as isize, j as isize, i as isize, 3);
    let wrk1 = Linear3D::new(wrk1, k as isize, j as isize, i as isize);
    let mut wrk2 = Linear3D::new(wrk2, k as isize, j as isize, i as isize);
    let bnd = Linear3D::new(bnd, k as isize, j as isize, i as isize);

    // iterate over necessary blocks in x direction (k direction)
    let mut bid_x = _block_idx_x();
    // during iteration we need to check if we are still in the domain by
    // using the thread id in x in the grid. We add the number of threads that
    // are present in x in the grid after each iteration
    let mut gtid_x_tmp = gtid_x;
    while bid_x < nblocks_x {
        // iterate over necessary blocks in y direction (j direction)
        let mut bid_y = _block_idx_y();
        // during iteration we need to check if we are still in the domain by
        // using the thread id in y in the grid. We add the number of threads that
        // are present in y in the grid after each iteration
        let mut gtid_y_tmp = gtid_y;
        while bid_y < nblocks_y {
            let mut z = 0;
            // load bottom plane
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_bot.set(p.get(gtid_x_tmp, gtid_y_tmp, z), tid_x, tid_y);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_bot.set(p.get(gtid_x_tmp + 2, gtid_y_tmp, z), tid_x + 2, tid_y);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_bot.set(p.get(gtid_x_tmp, gtid_y_tmp + 2, z), tid_x, tid_y + 2);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_bot.set(
                    p.get(gtid_x_tmp + 2, gtid_y_tmp + 2, z),
                    tid_x + 2,
                    tid_y + 2,
                );
            }
            // load mid plane
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_mid.set(p.get(gtid_x_tmp, gtid_y_tmp, z + 1), tid_x, tid_y);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_mid.set(p.get(gtid_x_tmp + 2, gtid_y_tmp, z + 1), tid_x + 2, tid_y);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_mid.set(p.get(gtid_x_tmp, gtid_y_tmp + 2, z + 1), tid_x, tid_y + 2);
            }
            _syncthreads();
            if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                p_sh_mid.set(
                    p.get(gtid_x_tmp + 2, gtid_y_tmp + 2, z + 1),
                    tid_x + 2,
                    tid_y + 2,
                );
            }
            // iterate in i direction
            while z < i - 2 {
                // load top plane
                if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                    p_sh_top.set(p.get(gtid_x_tmp, gtid_y_tmp, z + 2), tid_x, tid_y);
                }
                _syncthreads();
                if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                    p_sh_top.set(p.get(gtid_x_tmp + 2, gtid_y_tmp, z + 2), tid_x + 2, tid_y);
                }
                _syncthreads();
                if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                    p_sh_top.set(p.get(gtid_x_tmp, gtid_y_tmp + 2, z + 2), tid_x, tid_y + 2);
                }
                _syncthreads();
                if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                    p_sh_top.set(
                        p.get(gtid_x_tmp + 2, gtid_y_tmp + 2, z + 2),
                        tid_x + 2,
                        tid_y + 2,
                    );
                }
                _syncthreads();
                if gtid_x_tmp < k - 2 && gtid_y_tmp < j - 2 {
                    // we need those as isize so we shadow them
                    let gtid_x_tmp = gtid_x_tmp as isize;
                    let gtid_y_tmp = gtid_y_tmp as isize;
                    // coefficients are loaded for the index we calculate
                    let a0 = a.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 0);
                    let a1 = a.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 1);
                    let a2 = a.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 2);
                    let a3 = a.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 3);
                    let b0 = b.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 0);
                    let b1 = b.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 1);
                    let b2 = b.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 2);
                    let c0 = c.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 0);
                    let c1 = c.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 1);
                    let c2 = c.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1, 2);
                    let bnd = bnd.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1);
                    let wrk1 = wrk1.get(gtid_x_tmp + 1, gtid_y_tmp + 1, z as isize + 1);
                    // do iterative jacobi
                    let s0 = a0 * p_sh_top.get(tid_x + 1, tid_y + 1)
                        + a1 * p_sh_mid.get(tid_x + 1, tid_y + 2)
                        + a2 * p_sh_mid.get(tid_x + 2, tid_y + 1)
                        + b0 * (p_sh_top.get(tid_x + 1, tid_y + 2)
                            - p_sh_top.get(tid_x + 1, tid_y)
                            - p_sh_bot.get(tid_x + 1, tid_y + 2)
                            + p_sh_bot.get(tid_x + 1, tid_y))
                        + b1 * (p_sh_mid.get(tid_x + 2, tid_y + 2)
                            - p_sh_mid.get(tid_x, tid_y + 2)
                            - p_sh_mid.get(tid_x + 2, tid_y)
                            + p_sh_mid.get(tid_x, tid_y))
                        + b2 * (p_sh_top.get(tid_x + 2, tid_y + 1)
                            - p_sh_top.get(tid_x, tid_y + 1)
                            - p_sh_bot.get(tid_x + 2, tid_y + 1)
                            + p_sh_bot.get(tid_x, tid_y + 1))
                        + c0 * p_sh_bot.get(tid_x + 1, tid_y + 1)
                        + c1 * p_sh_mid.get(tid_x + 1, tid_y)
                        + c2 * p_sh_mid.get(tid_x, tid_y + 1)
                        + wrk1;
                    let ss = (s0 * a3 - p_sh_mid.get(tid_x + 1, tid_y + 1)) * bnd;
                    wrk2.set(
                        p_sh_mid.get(tid_x + 1, tid_y + 1) + omega * ss,
                        gtid_x_tmp + 1,
                        gtid_y_tmp + 1,
                        z as isize + 1,
                    );
                    residual += ss * ss;
                }
                // swap smem planes
                let tmp = p_sh_bot;
                p_sh_bot = p_sh_mid;
                p_sh_mid = p_sh_top;
                p_sh_top = tmp;
                _syncthreads();
                z += 1;
            }
            bid_y += _grid_dim_y();
            gtid_y_tmp += gnthreads_y;
        }
        bid_x += _grid_dim_x();
        gtid_x_tmp += gnthreads_x;
    }

    p_sh_top.set(residual, tid, 0);
    _syncthreads();
    // reduce the residual of each thread block
    let mut nworkers = get_init_worker_count(nthreads);
    while nworkers > 0 {
        if tid < nworkers && (tid + nworkers) < nthreads {
            let my = p_sh_top.get(tid, 0);
            let other = p_sh_top.get(tid + nworkers, 0);
            p_sh_top.set(my + other, tid, 0);
        }
        _syncthreads();
        nworkers /= 2;
    }
    // writing final result back atomically
    if tid_x == 0 && tid_y == 0 {
        _atomic_add_f32(gosa, p_sh_top.get(tid, 0));
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn add_without_wrap(a: i32, b: i32, c: *mut i32) {
    let thread_id = _thread_idx_x() + _block_dim_x() * _block_idx_x();
    *c.offset(thread_id as isize) = a + b;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn add_wrap(a: i32, b: i32, c: *mut i32) {
    let thread_id = _thread_idx_x() + _block_dim_x() * _block_idx_x();
    *c.offset(thread_id as isize) = a.wrapping_add(b);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_test(input: *const f32, output: *mut f32) {
    let gtid_x = _block_idx_x() * _grid_dim_x() + _thread_idx_x();
    _atomic_add_f32(output, *input.offset(gtid_x as isize) * 2f32);
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
