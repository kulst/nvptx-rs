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
    // thread_idx_x in block
    let tid_x = _thread_idx_x() as isize;
    // thread_idx_y in block
    let tid_y = _thread_idx_y() as isize;
    // calculate local thread id with 2D thread blocks tid.y * block_dim.x + tid.x
    let thread_id_local = tid_y * _block_dim_x() as isize + tid_x;
    // calculate local thread count with 2D thread blocks block_dim.x * block_dim.y
    let thread_count_local = (_block_dim_x() * _block_dim_y()) as isize;
    // calculate global thread id in x direction
    let thread_idx_global = _block_dim_x() * _block_idx_x() + _thread_idx_x();
    // calculate global number of threads in x direction
    let thread_count_x_global = _block_dim_x() * _grid_dim_x();
    // calculate global thread id in x direction
    let thread_idy_global = _block_dim_y() * _block_idx_y() + _thread_idx_y();
    // global number of threads in y direction
    let thread_count_y_global = _block_dim_y() * _grid_dim_y();

    let blocks_x = (k - 2 + _block_dim_x() - 1) / _block_dim_x();
    let blocks_y = (j - 2 + _block_dim_y() - 1) / _block_dim_y();

    let mut residual = 0f64;

    let mut p = TexObjectF32_3D::new(p, k, j, i);

    let smem_rows = (_block_dim_y() + 2) as isize;
    let smem_cols = (_block_dim_x() + 2) as isize;

    let smem_len = (smem_rows * smem_cols) as usize;

    let p_sh_top = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_top = Linear2D::new(p_sh_top, smem_cols, smem_rows);
    let p_sh_mid = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_mid = Linear2D::new(p_sh_mid, smem_cols, smem_rows);
    let p_sh_bot = _dynamic_shared_mem::<f32>(smem_len);
    let mut p_sh_bot = Linear2D::new(p_sh_bot, smem_cols, smem_rows);
    let a = Linear4D::new(a, k as isize, j as isize, i as isize, 4);
    let b = Linear4D::new(b, k as isize, j as isize, i as isize, 3);
    let c = Linear4D::new(c, k as isize, j as isize, i as isize, 3);
    let wrk1 = Linear3D::new(wrk1, k as isize, j as isize, i as isize);
    let mut wrk2 = Linear3D::new(wrk2, k as isize, j as isize, i as isize);
    let bnd = Linear3D::new(bnd, k as isize, j as isize, i as isize);

    let p_sh_top_tmp = p_sh_top.clone();

    // iterate in k direction
    let mut bx = _block_idx_x();
    let mut thread_idx_global_tmp = thread_idx_global;
    while bx < blocks_x {
        // iterate in j direction
        let mut by = _block_idx_y();
        let mut thread_idy_global_tmp = thread_idy_global;
        while by < blocks_y {
            let mut z = 0;
            // load bottom plane
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_bot.set(
                    p.get(thread_idx_global_tmp, thread_idy_global_tmp, z),
                    tid_x,
                    tid_y,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_bot.set(
                    p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp, z),
                    tid_x + 2,
                    tid_y,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_bot.set(
                    p.get(thread_idx_global_tmp, thread_idy_global_tmp + 2, z),
                    tid_x,
                    tid_y + 2,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_bot.set(
                    p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp + 2, z),
                    tid_x + 2,
                    tid_y + 2,
                );
            }
            _syncthreads();
            // load mid plane
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_mid.set(
                    p.get(thread_idx_global_tmp, thread_idy_global_tmp, z + 1),
                    tid_x,
                    tid_y,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_mid.set(
                    p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp, z + 1),
                    tid_x + 2,
                    tid_y,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_mid.set(
                    p.get(thread_idx_global_tmp, thread_idy_global_tmp + 2, z + 1),
                    tid_x,
                    tid_y + 2,
                );
            }
            _syncthreads();
            if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                p_sh_mid.set(
                    p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp + 2, z + 1),
                    tid_x + 2,
                    tid_y + 2,
                );
            }
            // iterate in i direction
            while z < i - 2 {
                _syncthreads();
                // load top plane
                if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                    p_sh_top.set(
                        p.get(thread_idx_global_tmp, thread_idy_global_tmp, z + 2),
                        tid_x,
                        tid_y,
                    );
                }
                _syncthreads();
                if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                    p_sh_top.set(
                        p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp, z + 2),
                        tid_x + 2,
                        tid_y,
                    );
                }
                _syncthreads();
                if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                    p_sh_top.set(
                        p.get(thread_idx_global_tmp, thread_idy_global_tmp + 2, z + 2),
                        tid_x,
                        tid_y + 2,
                    );
                }
                _syncthreads();
                if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                    p_sh_top.set(
                        p.get(thread_idx_global_tmp + 2, thread_idy_global_tmp + 2, z + 2),
                        tid_x + 2,
                        tid_y + 2,
                    );
                }
                _syncthreads();
                if thread_idx_global_tmp < k - 2 && thread_idy_global_tmp < j - 2 {
                    let thread_idx_global_tmp = thread_idx_global_tmp as isize;
                    let thread_idy_global_tmp = thread_idy_global_tmp as isize;

                    let a0 = a.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        0,
                    );
                    let a1 = a.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        1,
                    );
                    let a2 = a.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        2,
                    );
                    let a3 = a.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        3,
                    );
                    let b0 = b.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        0,
                    );
                    let b1 = b.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        1,
                    );
                    let b2 = b.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        2,
                    );
                    let c0 = c.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        0,
                    );
                    let c1 = c.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        1,
                    );
                    let c2 = c.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                        2,
                    );
                    let bnd = bnd.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                    );
                    let wrk1 = wrk1.get(
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                    );
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
                        thread_idx_global_tmp + 1,
                        thread_idy_global_tmp + 1,
                        z as isize + 1,
                    );

                    residual += ss as f64 * ss as f64;

                    #[repr(C)]
                    struct P(
                        i32,
                        i32,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                    );
                    if thread_idx_global == 0 && thread_idy_global == 0 {
                        core::arch::nvptx::vprintf(
                            "gidx: %d, gidy: %d, residual %f, \
                        ss: %f, a0: %f, a1: %f, a2: %f, \
                        a3: %f, b0: %f, b1: %f, b2: %f, \
                        c0: %f, c1: %f, c2: %f, bnd: %f, wrk1: %f\n"
                                .as_ptr(),
                            core::mem::transmute(&P(
                                thread_idx_global_tmp as i32,
                                thread_idy_global_tmp as i32,
                                residual.into(),
                                p_sh_top.get(tid_x + 1, tid_y + 1).into(),
                                p_sh_mid.get(tid_x + 1, tid_y + 2).into(),
                                p_sh_mid.get(tid_x + 2, tid_y + 1).into(),
                                p_sh_top.get(tid_x + 1, tid_y + 2).into(),
                                p_sh_top.get(tid_x + 1, tid_y).into(),
                                p_sh_bot.get(tid_x + 1, tid_y + 2).into(),
                                p_sh_bot.get(tid_x + 1, tid_y).into(),
                                p_sh_mid.get(tid_x + 2, tid_y + 2).into(),
                                p_sh_mid.get(tid_x, tid_y + 2).into(),
                                p_sh_mid.get(tid_x + 2, tid_y).into(),
                                p_sh_mid.get(tid_x, tid_y).into(),
                                p_sh_top.get(tid_x + 2, tid_y + 1).into(),
                                p_sh_top.get(tid_x, tid_y + 1).into(),
                                p_sh_bot.get(tid_x + 2, tid_y + 1).into(),
                                p_sh_bot.get(tid_x, tid_y + 1).into(),
                                p_sh_bot.get(tid_x + 1, tid_y + 1).into(),
                                p_sh_mid.get(tid_x + 1, tid_y).into(),
                                p_sh_mid.get(tid_x, tid_y + 1).into(),
                            )),
                        );
                    }
                }
                // swap smem planes
                let tmp = p_sh_bot;
                p_sh_bot = p_sh_mid;
                p_sh_mid = p_sh_top;
                p_sh_top = tmp;
                _syncthreads();
                z += 1;
            }
            by += _grid_dim_y();
            thread_idy_global_tmp += thread_count_y_global;
        }
        bx += _grid_dim_x();
        thread_idx_global_tmp += thread_count_x_global;
    }

    _syncthreads();
    // #[repr(C)]
    // struct P(i32, i32, f64);
    // core::arch::nvptx::vprintf(
    //     "gidx: %d, gidy: %d, residual %f\n".as_ptr(),
    //     core::mem::transmute(&P(thread_idx_global, thread_idy_global, residual.into())),
    // );
    // _atomic_add_f32(gosa, residual as f32);
    // now write the residual of each thread block into the top plane in shared memory
    // let mut p_sh_top = p_sh_top_tmp;
    p_sh_top.set(residual as f32, thread_id_local, 0);
    _syncthreads();
    // now reduce the residual of each thread block
    let mut workers = get_init_worker_count(thread_count_local);
    while workers > 0 {
        if thread_id_local < workers && (thread_id_local + workers) < thread_count_local {
            let my = p_sh_top.get(thread_id_local, 0);
            let other = p_sh_top.get(thread_id_local + workers, 0);
            p_sh_top.set(my + other, thread_id_local, 0);
        }
        _syncthreads();
        workers /= 2;
    }
    if tid_x == 0 && tid_y == 0 {
        _atomic_add_f32(gosa, p_sh_top.get(thread_id_local, 0));
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

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
