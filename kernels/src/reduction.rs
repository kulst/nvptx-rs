use core::arch::nvptx::vprintf;
use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::mem::transmute;
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
use core::ops::AddAssign;

#[inline]
pub(crate) unsafe fn reduction<T>(input: *const T, output: *mut T, n: usize)
where
    T: FloatCore + 'static + FromPrimitive + AddAssign,
{
    // initialize dynamic shared memory
    let mut dyn_smem = DynSmem::new();
    let input = SimtArray::<_, Grid, Init, Dense, _>::new_unchecked(
        input as *mut T,
        DenseMapping::new((n,)),
    );
    let input_view = input.view::<Thread, (Xyz,)>();
    //
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
    let input_view = input_view.map(|input_view| {
        input_view.into_view(StepRange {
            range: 1..,
            step: 1,
        })
    });
    {
        let mut local_sum_view = local_sum.view_mut::<Thread, (Xyz,)>();
        if let Some((local_val, input)) =
            local_sum_view.expr_mut().as_mut().zip(input_view.as_ref())
        {
            for_each(input, |input_val| local_val[0] += *input_val)
        }
    }
    // // reduce the local_sum
    let mut nworkers = get_init_worker_count(n_local);
    while nworkers > 0 {
        let mut local_sum_view =
            local_sum.view_mut_with_limited_quantity::<Thread, (Xyz,)>(nworkers as u32);
        if let Some(mut local_sum_view) = local_sum_view.expr_mut() {
            if local_sum_view.dim(0) >= 2 {
                let other = local_sum_view[1];
                local_sum_view[0] += other;
            }
        }
        nworkers /= 2;
    }
    //
    let local_sum_view = local_sum.expr();
    // Write block result back into global memory
    if <Xyz as Projection<Thread, Block>>::idx() == 0 {
        _atomic_add::<T>(output, local_sum_view[0]);
    }
}
