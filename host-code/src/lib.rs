use num::traits::float::FloatCore;
use rayon::prelude::*;
use std::ops::{Index, IndexMut};

/// To calculate the stencil on the host for value validation, we create a helper
/// struct for easier handling of 3D matrices
#[derive(Clone)]
pub struct Matrix<T> {
    buffer: Vec<T>,
    cols: usize,
    rows: usize,
    deps: usize,
}

impl<T> Matrix<T>
where
    T: FloatCore + 'static,
{
    pub fn from_vec(vec: Vec<T>, cols: usize, rows: usize, deps: usize) -> Result<Self, ()> {
        if vec.len() != cols * rows * deps {
            return Err(());
        }
        Ok(Self {
            buffer: vec,
            cols,
            rows,
            deps,
        })
    }

    pub fn clone_zeroed(&self) -> Self {
        Self {
            buffer: vec![T::zero(); self.cols * self.rows * self.deps],
            cols: self.cols,
            rows: self.rows,
            deps: self.deps,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }

    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_slice()
    }

    pub fn size(&self) -> usize {
        self.deps * self.rows * self.cols
    }

    pub fn get_cols(&self) -> usize {
        self.cols
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_deps(&self) -> usize {
        self.deps
    }
    // fn get_index(&self, index: usize) -> Option<(usize, usize, usize)> {
    //     if index >= self.size() {
    //         return None;
    //     }
    //     let col = index % self.cols;
    //     let row = (index / self.cols) % self.rows;
    //     let dep = index / (self.cols * self.rows);
    //     Some((dep, row, col))
    // }

    // fn is_edge(&self, index: usize) -> Option<bool> {
    //     if let Some((col, row, dep)) = self.get_index(index) {
    //         if col == 0
    //             || row == 0
    //             || dep == 0
    //             || col == self.cols - 1
    //             || row == self.rows - 1
    //             || dep == self.deps - 1
    //         {
    //             return Some(true);
    //         }
    //         return Some(false);
    //     }
    //     None
    // }

    pub fn into_buffer(self) -> Vec<T> {
        self.buffer
    }
}

impl<T> Index<(usize, usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.buffer[index.2 + index.1 * self.cols + index.0 * self.cols * self.rows]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.buffer[index.2 + index.1 * self.cols + index.0 * self.cols * self.rows]
    }
}

pub fn himeno_stencil<T>(
    input: &Matrix<T>,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
    b: T,
    c: T,
    wrk1: T,
    bnd: T,
    omega: T,
) -> Matrix<T>
where
    T: Clone + FloatCore + 'static + Send + Sync,
{
    let (b0, b1, b2, c0, c1, c2) = (b, b, b, c, c, c);
    let mut output = input.clone_zeroed();
    let i_max = input.deps;
    let j_max = input.rows;
    let k_max = input.cols;

    output
        .buffer
        .par_chunks_mut(j_max * k_max)
        .enumerate()
        .for_each(|(i, slice)| {
            slice
                .par_chunks_mut(k_max)
                .enumerate()
                .for_each(|(j, slice)| {
                    slice.iter_mut().enumerate().for_each(|(k, slice)| {
                        *slice = if i > 0usize
                            && j > 0usize
                            && k > 0usize
                            && i < i_max - 1
                            && j < j_max - 1
                            && k < k_max - 1
                        {
                            let s0 = a0 * input[(i + 1, j, k)]
                                + a1 * input[(i, j + 1, k)]
                                + a2 * input[(i, j, k + 1)]
                                + b0 * (input[(i + 1, j + 1, k)]
                                    - input[(i + 1, j - 1, k)]
                                    - input[(i - 1, j + 1, k)]
                                    + input[(i - 1, j - 1, k)])
                                + b1 * (input[(i, j + 1, k + 1)]
                                    - input[(i, j - 1, k + 1)]
                                    - input[(i, j + 1, k - 1)]
                                    + input[(i, j - 1, k - 1)])
                                + b2 * (input[(i + 1, j, k + 1)]
                                    - input[(i - 1, j, k + 1)]
                                    - input[(i + 1, j, k - 1)]
                                    + input[(i - 1, j, k - 1)])
                                + c0 * input[(i - 1, j, k)]
                                + c1 * input[(i, j - 1, k)]
                                + c2 * input[(i, j, k - 1)]
                                + wrk1;
                            let ss = s0 * a3 - input[(i, j, k)] * bnd;

                            input[(i, j, k)] + omega * ss
                        } else {
                            T::zero()
                        }
                    });
                });
        });
    output
}
