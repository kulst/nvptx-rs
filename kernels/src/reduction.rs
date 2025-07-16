use mdarray::expr::for_each;
use mdarray::Dense;
use mdarray::DenseMapping;
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
    // Create one one-dimensional SimtArray per Grid (it is the same in the
    // whole Grid) which is already initialized
    let input = SimtArray::<_, Grid, Init, Dense, _>::new_unchecked(
        input as *mut T,
        DenseMapping::new((n,)),
    );
    // Create a disjunct view into the input for each Thread determined by
    // the Xyz projection for the only dimension 0 (might be None for some threads
    // if the number of threads in the grid is larger than n)
    let input_view = input.view::<Thread, (Xyz,)>();
    // Create a thread-local value which reduces all values of the disjunct input
    // view of the thread
    let mut local_val = T::zero();
    if let Some(input_view) = input_view {
        for_each(input_view, |input_val| local_val += *input_val)
    }
    // n_local is the count of threads per block considering all three dimension
    // It is equivalent to _block_dim_x() * _block_dim_y() * _block_dim_z()
    let n_local = <Xyz as Projection<Thread, Block>>::dim().as_();
    // Create one one-dimensional uninitialized SimtArray in shared memory
    // per Block (it is different for each block) and initialize it with the
    // thread-local reduced value.
    // We do not have to explicitely synchronize as this is implicetly done
    // after the closure is called
    let mut local_sum = SimtArray::<_, Block, Uninit, Dense, _>::new_unchecked(
        DynSmem::get_chunk::<T>(&mut dyn_smem, n_local),
        DenseMapping::new((n_local,)),
    )
    .init_with::<Thread, (Xyz,), _>(|_| local_val);
    // calculating the number of threads which are necessary for the reduction
    // this is the next smallest power of 2
    let mut nworkers = get_init_worker_count(n_local);
    // As long as we still need workers..
    while nworkers > 0 {
        // For all workers we create a mutable view into the local_sum, the others
        // get None
        let mut local_sum_ref_mut =
            local_sum.view_mut_with_limited_quantity::<Thread, (Xyz,)>(nworkers as u32);
        // Only if we are a worker (the view is Some(..)) ..
        if let Some(mut local_sum_view) = local_sum_ref_mut.expr_mut() {
            // and we have at least 2 elements in the view
            if local_sum_view.dim(0) >= 2 {
                // we reduce these two elements into one
                let other = local_sum_view[1];
                local_sum_view[0] += other;
            }
        }
        // We do not have to explicitely synchronize here, as this is done
        // implicetly when dropping the local_sum_ref_mut after each loop iteration

        // We halve the number of workers
        nworkers /= 2;
    }
    // We create an immutable view into the local sum ..
    let local_sum_view = local_sum.expr();
    // and only if we are thread 0 inside the block we atomically add our
    // local_sum to the global sum
    if <Xyz as Projection<Thread, Block>>::idx() == 0 {
        _atomic_add::<T>(output, local_sum_view[0]);
    }
}
