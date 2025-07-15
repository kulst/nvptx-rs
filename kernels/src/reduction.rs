use core::cell::UnsafeCell;
use mdarray::expr::for_each;
use mdarray::expr::Apply;
use mdarray::Dense;
use mdarray::DenseMapping;
use mdarray::Dyn;
use mdarray::StepRange;
use mdarray::Strided;
use mdarray::StridedMapping;
use mdarray::View;
use mdarray::ViewMut;
use num::{traits::float::FloatCore, FromPrimitive};

use simtarray::nvptx::Block;
use simtarray::nvptx::Grid;
use simtarray::nvptx::Thread;
use simtarray::nvptx::Xyz;
use simtarray::*;

use crate::intrinsics::*;
use crate::util::*;
use core::hint::assert_unchecked;
use core::iter::repeat;
use core::mem::MaybeUninit;
use core::{arch::nvptx::*, ops::AddAssign};

#[inline]
pub(crate) unsafe fn reduction<T>(input: *const T, output: *mut T, n: usize)
where
    T: FloatCore + 'static + FromPrimitive + AddAssign,
{
    // initialize dynamic shared memory
    let mut dyn_smem = DynSmem::new();
    // // thread id in block
    // let tid = _thread_idx_x() as usize;
    // // thread id in grid
    // let gtid = (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as usize;
    // // number of threads in block
    // let nthreads = _block_dim_x() as usize;
    // // number of threads in grid
    // let gnthreads = (_block_dim_x() * _grid_dim_x()) as usize;
    // // We are sure that gnthreads is larger than 0 so we help the compiler a little bit
    // assert_unchecked(gnthreads > 0);
    // get chunk in dynamic shared memory

    let input = SimtArray::<_, Grid, Init, Dense, _>::new_unchecked(
        input as *mut T,
        DenseMapping::new((n,)),
    );
    let input_view = input.view::<Thread, (Xyz,)>();

    let n_local = <Xyz as Projection<Thread, Block>>::dim().as_();
    let local_sum = DynSmem::get_chunk::<T>(&mut dyn_smem, n_local);
    let local_sum = SimtArray::<_, Block, Uninit, Dense, _>::new_unchecked(
        local_sum,
        DenseMapping::new((n_local,)),
    );
    // Initialize local_sum with first input value or zero if gtid >= n
    let mut local_sum = local_sum.init_with::<Thread, (Xyz,), _>(|_| {
        if let Some(input_view) = input_view {
            input_view[0]
        } else {
            T::zero()
        }
    });
    let (local_sum_view, local_sum_token) = local_sum.view_mut::<Thread, (Xyz,)>();
    drop(local_sum_token);
    let local_sum_view = if let Some(local_sum_view) = local_sum_view {
        Some(local_sum_view.into_view(StepRange {
            range: 1..,
            step: 1,
        }))
    } else {
        None
    };
    // let mut local_sum_view: ViewMut<'_, _, _, Dense> =
    //     ViewMut::new_unchecked(local_ptr as *mut T, local_mapping);
    // // Add subsequent values to local_s hum
    // let input = if let Some(input) = input {
    //     if input.len() > 1 {
    //         Some(View::<'_, T, (usize,), _>::into_view(
    //             input,
    //             StepRange {
    //                 range: 1..,
    //                 step: 1,
    //             },
    //         ))
    //     } else {
    //         None
    //     }
    // } else {
    //     None
    // };
    // if let Some(input) = input {
    //     for_each(input, |input_val| {
    //         local_sum_view[0] += *input_val;
    //     });
    // }
    // _syncthreads();
    // // reduce the local_sum
    // let mut nworkers = get_init_worker_count(nthreads);
    // while nworkers > 0 {
    //     if tid < nworkers && (tid + nworkers) < nthreads {
    //         let local_sum_view: View<'_, _, (usize,), Dense> = View::new_unchecked(
    //             local_sum as *const UnsafeCell<T>,
    //             DenseMapping::new((nthreads,)),
    //         );
    //         let local_sum_view = View::<'_, _, (usize,), _>::into_view(
    //             local_sum_view,
    //             StepRange {
    //                 range: tid..,
    //                 step: nworkers as isize,
    //             },
    //         );
    //         let (local_ptr, local_mapping) = local_sum_view.into_raw_parts();
    //         let mut local_sum_view: ViewMut<'_, _, (usize,), Strided> =
    //             ViewMut::new_unchecked(local_ptr as *mut T, local_mapping);
    //         let other = local_sum_view[1];
    //         local_sum_view[0] += other;
    //     }
    //     _syncthreads();
    //     nworkers /= 2;
    // }
    // let local_sum_view: View<'_, _, _, Dense> =
    //     View::new_unchecked(local_ptr as *const T, DenseMapping::new((n,)));
    // // Write block result back into global memory
    // if tid == 0 {
    //     _atomic_add::<T>(output, local_sum_view[0]);
    // }
}
