use num::{traits::float::FloatCore, FromPrimitive};

use crate::intrinsics::*;
use core::{arch::nvptx::*, ops::AddAssign};

#[inline]
pub(crate) unsafe fn matrixmultiplication<T>(a: *const T, b: *const T, c: *mut T, n: usize)
where
    T: FloatCore + 'static + FromPrimitive + AddAssign,
{
    // compute global indices for current thread
    let row = (_block_dim_y() * _block_idx_y() + _thread_idx_y()) as usize;
    let col = (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as usize;
    let index = row * n + col;

    // Initialize local result
    let mut result = T::zero();
    if row >= n || col >= n {
        return;
    }

    // Compute matrix multiplication
    for i in 0..n {
        result += a.add(row * n + i).read() * b.add(i * n + col).read();
    }

    c.add(index).write(result);
}

pub(crate) unsafe fn matrixmultiplication_shared_memory<T>(
    a: *const T,
    b: *const T,
    c: *mut T,
    n: usize,
) where
    T: FloatCore + 'static + FromPrimitive + AddAssign,
{
    const TILE_SIZE: usize = 16;
    //initialize shared memory
    let shared = _static_shared_mem::<[T; TILE_SIZE * TILE_SIZE * 2]>();
    let shared_a: *mut T = shared.cast();
    let shared_b = shared_a.add(TILE_SIZE * TILE_SIZE);

    //Compute thread and block indices
    let tidx_x = _thread_idx_x() as usize;
    let tidx_y = _thread_idx_y() as usize;
    let tidx = tidx_y * _block_dim_x() as usize + tidx_x;
    let bidx_x = _block_idx_x() as usize;
    let bidx_y = _block_idx_y() as usize;

    //Calculate column and row indices
    let col = bidx_x * TILE_SIZE + tidx_x;
    let row = bidx_y * TILE_SIZE + tidx_y;

    if row >= n && col >= n {
        return;
    }

    // initialize result
    let mut result = T::zero();

    //calculate matrix multiplication
    for tile_idx in 0..n.div_ceil(TILE_SIZE) {
        //load next tile of matrix a into shared memory
        if row < n && (tile_idx * TILE_SIZE + tidx_x) < n {
            shared_a
                .add(tidx)
                .write(a.add(row * n + tile_idx * TILE_SIZE + tidx_x).read());
        } else {
            shared_a.add(tidx).write(T::zero());
        }

        //load next tile of matrix b into shared memory
        if col < n && (tile_idx * TILE_SIZE + tidx_y) < n {
            shared_b
                .add(tidx)
                .write(b.add((tile_idx * TILE_SIZE + tidx_y) * n + col).read());
        } else {
            shared_b.add(tidx).write(T::zero());
        }

        //wait until all tiles loaded
        _syncthreads();

        //multiply tiles
        for idx in 0..TILE_SIZE {
            result += shared_a.add(tidx_y * TILE_SIZE + idx).read()
                * shared_b.add(idx * TILE_SIZE + tidx_x).read()
        }

        // wait all threads finished calculation of result
        _syncthreads();
    }

    // Write final result into output matrix c
    if row < n && col < n {
        c.add(row * n + col).write(result);
    }
}
