use num::traits::float::FloatCore;
use num::FromPrimitive;

use crate::intrinsics::*;
use crate::linear::*;
use core::arch::nvptx::*;
use core::iter::successors;

#[inline]
pub(crate) unsafe fn stencil<T: FloatCore + 'static + FromPrimitive>(
    p: *const T,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
    b: T,
    c: T,
    wrk1: T,
    bnd: T,
    wrk2: *mut T,
    omega: T,
    i: i32,
    j: i32,
    k: i32,
) {
    // initialize dynamic shared memory
    let mut dyn_smem = DynSmem::new();
    // thread id x in block
    let tid_x = _thread_idx_x() as usize;
    // thread id y in block
    let tid_y = _thread_idx_y() as usize;
    // thread id x in grid
    let gtid_x = (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as usize;
    // number of threads in x in grid
    let gnthreads_x = (_block_dim_x() * _grid_dim_x()) as usize;
    // thread id y in grid
    let gtid_y = (_block_dim_y() * _block_idx_y() + _thread_idx_y()) as usize;
    // number of threads in y in grid
    let gnthreads_y = (_block_dim_y() * _grid_dim_y()) as usize;
    // number of blocks in x necessary to process the input
    let nblocks_x = (k - 2 + _block_dim_x() - 1) / _block_dim_x();
    // number of blocks in y necessary to process the input
    let nblocks_y = (j - 2 + _block_dim_y() - 1) / _block_dim_y();
    // we need i, j and k as usize so we typecast and shadow the old ones
    let (i, j, k) = (i as usize, j as usize, k as usize);
    // // associate p as a Linear3D
    let p = Linear3D::new(p, k, j, i);
    // shared memory rows for p_sh_[top,mid,bot]
    let smem_rows = (_block_dim_y() + 2) as usize;
    // shared memory columns for p_sh_[top,mid,bot]
    let smem_cols = (_block_dim_x() + 2) as usize;
    // number of items for p_sh_[top,mid,bot]
    let smem_len = (smem_rows * smem_cols) as usize;
    // associate p_sh_top as Linear2D in shared memory
    let p_sh_top = DynSmem::get_chunk(&mut dyn_smem, smem_len);
    let mut p_sh_top = Linear2D::new(p_sh_top, smem_cols, smem_rows);
    // associate p_sh_mid as Linear2D in shared memory
    let p_sh_mid = DynSmem::get_chunk(&mut dyn_smem, smem_len);
    let mut p_sh_mid = Linear2D::new(p_sh_mid, smem_cols, smem_rows);
    // associate p_sh_bot as Linear2D in shared memory
    let p_sh_bot = DynSmem::get_chunk(&mut dyn_smem, smem_len);
    let mut p_sh_bot = Linear2D::new(p_sh_bot, smem_cols, smem_rows);
    // associate wrk2 as Linear3D
    let mut wrk2 = Linear3D::new(wrk2, k, j, i);
    // iterate over necessary blocks in x direction (k direction)
    // during iteration we need to check if we are still in the domain by
    // using the thread id in x in the grid. We add the number of threads that
    // are present in x in the grid after each iteration
    for (_bid_x, gtid_x) in (_block_idx_x()..nblocks_x)
        .step_by(_grid_dim_x() as usize)
        .zip(successors(Some(gtid_x), |&id| Some(id + gnthreads_x)))
    {
        // iterate over necessary blocks in y direction (j direction)
        // during iteration we need to check if we are still in the domain by
        // using the thread id in y in the grid. We add the number of threads that
        // are present in y in the grid after each iteration
        for (_bid_y, gtid_y) in (_block_idx_y()..nblocks_y)
            .step_by(_grid_dim_y() as usize)
            .zip(successors(Some(gtid_y), |&id| Some(id + gnthreads_y)))
        {
            // load bottom and mid plane if in domain
            for (tid_x, gtid_x) in successors(Some((tid_x, gtid_x)), |(tid, gtid)| {
                Some((
                    tid + _block_dim_x() as usize,
                    gtid + _block_dim_x() as usize,
                ))
            })
            .take_while(|(tid, gtid)| *tid < _block_dim_x() as usize + 2 && *gtid < k)
            {
                for (tid_y, gtid_y) in successors(Some((tid_y, gtid_y)), |(tid, gtid)| {
                    Some((
                        tid + _block_dim_y() as usize,
                        gtid + _block_dim_y() as usize,
                    ))
                })
                .take_while(|(tid, gtid)| *tid < _block_dim_y() as usize + 2 && *gtid < j)
                {
                    p_sh_bot.set(p.get(gtid_x, gtid_y, 0), tid_x, tid_y);
                    p_sh_mid.set(p.get(gtid_x, gtid_y, 1), tid_x, tid_y);
                }
            }
            // iterate in i direction
            for z in 0..(i - 2) {
                // load top plane if in domain
                for (tid_x, gtid_x) in successors(Some((tid_x, gtid_x)), |(tid, gtid)| {
                    Some((
                        tid + _block_dim_x() as usize,
                        gtid + _block_dim_x() as usize,
                    ))
                })
                .take_while(|(tid, gtid)| *tid < _block_dim_x() as usize + 2 && *gtid < k)
                {
                    for (tid_y, gtid_y) in successors(Some((tid_y, gtid_y)), |(tid, gtid)| {
                        Some((
                            tid + _block_dim_y() as usize,
                            gtid + _block_dim_y() as usize,
                        ))
                    })
                    .take_while(|(tid, gtid)| *tid < _block_dim_y() as usize + 2 && *gtid < j)
                    {
                        p_sh_top.set(p.get(gtid_x, gtid_y, z + 2), tid_x, tid_y);
                    }
                }
                _syncthreads();
                // calculate stencil if in domain
                if gtid_x < k - 2 && gtid_y < j - 2 {
                    // coefficients are loaded for the index we calculate
                    let (b0, b1, b2) = (b, b, b);
                    let (c0, c1, c2) = (c, c, c);
                    // do one iterative jacobi step
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
                        gtid_x + 1,
                        gtid_y + 1,
                        z + 1,
                    );
                }
                // swap smem planes
                let tmp = p_sh_bot;
                p_sh_bot = p_sh_mid;
                p_sh_mid = p_sh_top;
                p_sh_top = tmp;
                _syncthreads();
            }
        }
    }
}
