use num::{traits::float::FloatCore, FromPrimitive};

use crate::intrinsics::*;
use crate::util::*;
use core::hint::assert_unchecked;
use core::{arch::nvptx::*, ops::AddAssign};


#[inline]
pub(crate) unsafe fn reduction<T>(input: *const T, output: *mut T, n: usize)
where
    T: FloatCore + 'static + FromPrimitive + AddAssign,
{
    // initialize dynamic shared memory
    let mut dyn_smem = DynSmem::new();
    // thread id in block
    let tid = _thread_idx_x() as usize;
    // thread id in grid
    let gtid = (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as usize;
    // number of threads in block
    let nthreads = _block_dim_x() as usize;
    // number of threads in grid
    let gnthreads = (_block_dim_x() * _grid_dim_x()) as usize;
    // We are sure that gnthreads is larger than 0 so we help the compiler a little bit
    assert_unchecked(gnthreads > 0);
    // get chunk in dynamic shared memory
    let local_sum = DynSmem::get_chunk::<T>(&mut dyn_smem, nthreads);
    // Initialize local sum
    *local_sum.add(tid) = T::zero();
    // Fill local sum and do first reduction steps
    for i in (gtid..n).step_by(gnthreads) {
        *local_sum.add(tid) += *input.add(i);
    }
    _syncthreads();
    // reduce the local_sum
    let mut nworkers = get_init_worker_count(nthreads);
    while nworkers > 0 {
        if tid < nworkers && (tid + nworkers) < nthreads {
            let my = *local_sum.add(tid);
            let other = *local_sum.add(tid + nworkers);
            *local_sum.add(tid) = my + other;
        }
        _syncthreads();
        nworkers /= 2;
    }
    // Write block result back into global memory
    if tid == 0 {
        _atomic_add::<T>(output, *local_sum);
    }
}
