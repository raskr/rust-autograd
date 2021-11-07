/// Some gemm kernel usages are ported from ndarray
use crate::ndarray_ext::NdArray;
use crate::same_type;
use crate::tensor::Tensor;
#[cfg(feature = "blas")]
use crate::tensor_ops::blas_ffi::*;
use crate::Float;
use crate::NdArrayView;
use crate::{op, NdArrayViewMut};
use ndarray;
#[cfg(feature = "blas")]
use ndarray::Dimension;
use ndarray::{ArrayView2, ArrayViewMut2};
#[cfg(feature = "blas")]
use std::cmp;
#[cfg(feature = "blas")]
use std::mem;

#[cfg(feature = "blas")]
#[inline]
fn blas_row_major_2d<T: 'static, F>(a: &ndarray::ArrayView2<F>) -> bool
where
    F: Float,
{
    if !same_type::<F, T>() {
        return false;
    }
    is_blas_2d(&a.raw_dim(), a.strides(), MemoryOrder::C)
}

#[cfg(feature = "blas")]
#[inline]
fn blas_row_major_nd<T: 'static, F>(a: &NdArrayView<F>) -> bool
where
    F: Float,
{
    if !same_type::<F, T>() {
        return false;
    }
    let strides = a.strides();
    let rank = strides.len();
    is_blas_nd(
        a.shape(),
        strides[rank - 2],
        strides[rank - 1],
        MemoryOrder::C,
    )
}

#[cfg(feature = "blas")]
#[inline]
fn blas_row_major_2d_mut<T: 'static, F>(a: &ndarray::ArrayViewMut2<F>) -> bool
where
    F: Float,
{
    if !same_type::<F, T>() {
        return false;
    }
    is_blas_2d(&a.raw_dim(), a.strides(), MemoryOrder::C)
}

#[cfg(feature = "blas")]
#[inline]
fn blas_row_major_nd_mut<T: 'static, F>(a: &NdArrayViewMut<F>) -> bool
where
    F: Float,
{
    if !same_type::<F, T>() {
        return false;
    }
    let strides = a.strides();
    let rank = strides.len();
    is_blas_nd(
        a.shape(),
        strides[rank - 2],
        strides[rank - 1],
        MemoryOrder::C,
    )
}

#[cfg(feature = "blas")]
fn is_blas_nd(shape: &[usize], stride0: isize, stride1: isize, order: MemoryOrder) -> bool {
    let (m, n) = (shape[0], shape[1]);
    let (inner_stride, outer_dim) = match order {
        MemoryOrder::C => (stride1, n),
        MemoryOrder::F => (stride0, m),
    };
    if !(inner_stride == 1 || outer_dim == 1) {
        return false;
    }
    if stride0 < 1 || stride1 < 1 {
        return false;
    }
    if (stride0 > BlasIF::max_value() as isize || stride0 < BlasIF::min_value() as isize)
        || (stride1 > BlasIF::max_value() as isize || stride1 < BlasIF::min_value() as isize)
    {
        return false;
    }
    if m > BlasIF::max_value() as usize || n > BlasIF::max_value() as usize {
        return false;
    }
    true
}

#[cfg(feature = "blas")]
fn is_blas_2d(dim: &ndarray::Ix2, stride: &[isize], order: MemoryOrder) -> bool {
    let (m, n) = dim.into_pattern();
    let s0 = stride[0] as isize;
    let s1 = stride[1] as isize;
    let (inner_stride, outer_dim) = match order {
        MemoryOrder::C => (s1, n),
        MemoryOrder::F => (s0, m),
    };
    if !(inner_stride == 1 || outer_dim == 1) {
        return false;
    }
    if s0 < 1 || s1 < 1 {
        return false;
    }
    if (s0 > BlasIF::max_value() as isize || s0 < BlasIF::min_value() as isize)
        || (s1 > BlasIF::max_value() as isize || s1 < BlasIF::min_value() as isize)
    {
        return false;
    }
    if m > BlasIF::max_value() as usize || n > BlasIF::max_value() as usize {
        return false;
    }
    true
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[inline]
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    assert!(same_type::<A, B>());
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

// blas version of ndarray's mat_mul_impl
#[cfg(feature = "blas")]
fn mat_mul_impl_blas<F: Float>(
    alpha: F,
    lhs: &ArrayView2<'_, F>,
    rhs: &ArrayView2<'_, F>,
    beta: F,
    c: &mut ArrayViewMut2<'_, F>,
) {
    const GEMM_BLAS_CUTOFF: usize = 7;

    // size cutoff for using BLAS
    let cut = GEMM_BLAS_CUTOFF;
    let ((mut m, a), (_, mut n)) = (lhs.dim(), rhs.dim());
    if !(m > cut || n > cut || a > cut) || !(same_type::<F, f32>() || same_type::<F, f64>()) {
        return mat_mul_impl_slow(alpha, lhs, rhs, beta, c);
    }
    {
        // Use `c` for c-order and `f` for an f-order matrix
        // We can handle c * c, f * f generally and
        // c * f and f * c if the `f` matrix is square.
        let mut lhs_ = lhs.view();
        let mut rhs_ = rhs.view();
        let mut c_ = c.view_mut();
        let lhs_s0 = lhs_.strides()[0];
        let rhs_s0 = rhs_.strides()[0];
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;
        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
            mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_ = lhs_.reversed_axes();
            lhs_trans = CblasTrans
        } else if rhs_s0 == 1 && a == n {
            rhs_ = rhs_.reversed_axes();
            rhs_trans = CblasTrans
        }

        macro_rules! call_kernel_def {
            ($ty:ty, $f:ident) => {
                if blas_row_major_2d::<$ty, _>(&lhs_)
                    && blas_row_major_2d::<$ty, _>(&rhs_)
                    && blas_row_major_2d_mut::<$ty, _>(&c_)
                {
                    let (m, k) = match lhs_trans {
                        CblasNoTrans => lhs_.dim(),
                        _ => {
                            let (rows, cols) = lhs_.dim();
                            (cols, rows)
                        }
                    };
                    let n = match rhs_trans {
                        CblasNoTrans => rhs_.raw_dim()[1],
                        _ => rhs_.raw_dim()[0],
                    };
                    // adjust strides, these may [1, 1] for column matrices
                    let lhs_stride = cmp::max(lhs_.strides()[0] as BlasIF, k as BlasIF);
                    let rhs_stride = cmp::max(rhs_.strides()[0] as BlasIF, n as BlasIF);
                    let c_stride = cmp::max(c_.strides()[0] as BlasIF, n as BlasIF);

                    // gemm is C ← αA^Op B^Op + βC
                    // Where Op is notrans/trans/conjtrans
                    unsafe {
                        $f(
                            CblasRowMajor,
                            lhs_trans,
                            rhs_trans,
                            m as BlasIF,               // m, rows of Op(a)
                            n as BlasIF,               // n, cols of Op(b)
                            k as BlasIF,               // k, cols of Op(a)
                            cast_as(&alpha),           // alpha
                            lhs_.as_ptr() as *const _, // a
                            lhs_stride,                // lda
                            rhs_.as_ptr() as *const _, // b
                            rhs_stride,                // ldb
                            cast_as(&beta),            // beta
                            c_.as_mut_ptr() as *mut _, // c
                            c_stride,                  // ldc
                        );
                    }
                    return;
                }
            };
        }
        call_kernel_def!(f32, cblas_sgemm);
        call_kernel_def!(f64, cblas_dgemm);
    }
    mat_mul_impl_slow(alpha, lhs, rhs, beta, c)
}

#[allow(unused_assignments)]
#[cfg(feature = "blas")]
fn batch_mat_mul_impl_fast<F: Float>(
    alpha: F,
    lhs: &NdArrayView<'_, F>,
    rhs: &NdArrayView<'_, F>,
    beta: F,
    c: &mut NdArrayViewMut<'_, F>,
) {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let c_shape = c.shape();

    let rank = lhs.ndim();

    let (mut m, a, mut n) = (
        lhs_shape[rank - 2],
        lhs_shape[rank - 1],
        rhs_shape[rank - 1],
    );

    let lhs_batch_size: usize = lhs_shape[rank - 2..].iter().product();
    let rhs_batch_size: usize = rhs_shape[rank - 2..].iter().product();
    let c_batch_size: usize = c_shape[rank - 2..].iter().product();

    {
        use rayon::prelude::*;
        use std::slice;

        // Use `c` for c-order and `f` for an f-order matrix
        // We can handle c * c, f * f generally and
        // c * f and f * c if the `f` matrix is square.
        let mut lhs_ = lhs.view();
        let mut rhs_ = rhs.view();
        let mut c_ = c.view_mut();

        let mut lhs_strides = lhs_.strides();
        let mut rhs_strides = rhs_.strides();

        // copy if batch dims appear in last two dims.
        let copied_lhs;
        let copied_rhs;
        if batch_mat_mul_requires_copy(lhs_strides) {
            copied_lhs = crate::ndarray_ext::deep_copy(&lhs_);
            lhs_ = copied_lhs.view();
            lhs_strides = lhs_.strides();
        }
        if batch_mat_mul_requires_copy(rhs_strides) {
            copied_rhs = crate::ndarray_ext::deep_copy(&rhs_);
            rhs_ = copied_rhs.view();
            rhs_strides = rhs_.strides();
        }

        let lhs_s0 = lhs_strides[rank - 2];
        let rhs_s0 = rhs_strides[rank - 2];
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;

        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;

        // Update lhs, rhs info if needed
        if both_f {
            // A^t B^t = C^t => B A = C
            let mut lhs_t = lhs_;
            lhs_t.swap_axes(rank - 2, rank - 1);
            lhs_ = rhs_;
            lhs_.swap_axes(rank - 2, rank - 1);
            rhs_ = lhs_t;
            c_.swap_axes(rank - 2, rank - 1);
            mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_.swap_axes(rank - 2, rank - 1);
            lhs_trans = CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_.swap_axes(rank - 2, rank - 1);
            rhs_trans = CblasTrans;
        }

        #[cfg(feature = "blas")]
        {
            let lhs_slice = unsafe { slice::from_raw_parts(lhs_.as_ptr(), lhs_.len()) };
            let rhs_slice = unsafe { slice::from_raw_parts(rhs_.as_ptr(), rhs_.len()) };
            let c_slice = unsafe { slice::from_raw_parts_mut(c_.as_mut_ptr(), c_.len()) };

            macro_rules! call_kernel_def {
                ($ty:ty, $f:ident) => {
                    if blas_row_major_nd::<$ty, _>(&lhs_)
                        && blas_row_major_nd::<$ty, _>(&rhs_)
                        && blas_row_major_nd_mut::<$ty, _>(&c_)
                    {
                        let (m, k) = match lhs_trans {
                            CblasNoTrans => {
                                let s = lhs_.shape();
                                (s[rank - 2], s[rank - 1])
                            }
                            _ => {
                                let s = lhs_.shape();
                                (s[rank - 1], s[rank - 2])
                            }
                        };
                        let n = match rhs_trans {
                            CblasNoTrans => rhs_.raw_dim()[rank - 1],
                            _ => rhs_.raw_dim()[rank - 2],
                        };
                        // adjust strides, these may [1, 1] for column matrices
                        let lhs_stride = cmp::max(lhs_.strides()[rank - 2] as BlasIF, k as BlasIF);
                        let rhs_stride = cmp::max(rhs_.strides()[rank - 2] as BlasIF, n as BlasIF);
                        let c_stride = cmp::max(c_.strides()[rank - 2] as BlasIF, n as BlasIF);

                        let a = lhs_slice.par_iter().step_by(lhs_batch_size);
                        let b = rhs_slice.par_iter().step_by(rhs_batch_size);
                        let c = c_slice.par_iter_mut().step_by(c_batch_size);

                        a.zip_eq(b).zip_eq(c).for_each(|((lhs, rhs), c)| {
                            unsafe {
                                // blas
                                $f(
                                    CblasRowMajor,
                                    lhs_trans,
                                    rhs_trans,
                                    m as BlasIF,                 // m, rows of Op(a)
                                    n as BlasIF,                 // n, cols of Op(b)
                                    k as BlasIF,                 // k, cols of Op(a)
                                    cast_as(&alpha),             // alpha
                                    lhs as *const F as *const _, // a
                                    lhs_stride,                  // lda
                                    rhs as *const F as *const _, // b
                                    rhs_stride,                  // ldb
                                    cast_as(&beta),              // beta
                                    c as *mut F as *mut _,       // c
                                    c_stride,                    // ldc
                                );
                            }
                        });

                        return;
                    }
                };
            }
            call_kernel_def!(f32, cblas_sgemm);
            call_kernel_def!(f64, cblas_dgemm);
        }
    }
    batch_mat_mul_impl_slow(alpha, lhs, rhs, beta, c)
}

/// C ← α A B + β C
fn mat_mul_impl_slow<F: Float>(
    alpha: F,
    lhs: &ArrayView2<'_, F>,
    rhs: &ArrayView2<'_, F>,
    beta: F,
    c: &mut ArrayViewMut2<'_, F>,
) {
    let ((m, k), (_, n)) = (lhs.dim(), rhs.dim());
    // common parameters for gemm
    let ap = lhs.as_ptr();
    let bp = rhs.as_ptr();
    let cp = c.as_mut_ptr();
    let (rsc, csc) = (c.strides()[0], c.strides()[1]);
    macro_rules! kernel_call_def {
        ($ty:ty, $f:ident) => {
            if crate::same_type::<F, $ty>() {
                unsafe {
                    ::matrixmultiply::$f(
                        m,
                        k,
                        n,
                        cast_as(&alpha),
                        ap as *const _,
                        lhs.strides()[0],
                        lhs.strides()[1],
                        bp as *const _,
                        rhs.strides()[0],
                        rhs.strides()[1],
                        cast_as(&beta),
                        cp as *mut _,
                        rsc,
                        csc,
                    );
                }
            }
        };
    }
    kernel_call_def!(f32, sgemm);
    kernel_call_def!(f64, dgemm);
}

/// C ← α A B + β C
#[allow(unused_assignments)]
#[allow(unused)]
fn batch_mat_mul_impl_slow<F: Float>(
    alpha: F,
    lhs: &NdArrayView<'_, F>,
    rhs: &NdArrayView<'_, F>,
    beta: F,
    c: &mut NdArrayViewMut<'_, F>,
) {
    let mut lhs_ = lhs.view();
    let mut rhs_ = rhs.view();
    let c_ = c.view_mut();
    let mut lhs_strides = lhs_.strides();
    let mut rhs_strides = rhs_.strides();
    let rank = lhs_strides.len();

    let copied_lhs;
    let copied_rhs;
    {
        if batch_mat_mul_requires_copy(lhs_strides) {
            copied_lhs = crate::ndarray_ext::deep_copy(&lhs_);
            lhs_ = copied_lhs.view();
            lhs_strides = lhs_.strides();
        }
        if batch_mat_mul_requires_copy(rhs_strides) {
            copied_rhs = crate::ndarray_ext::deep_copy(&rhs_);
            rhs_ = copied_rhs.view();
            rhs_strides = rhs_.strides();
        }
    }

    let lhs_shape = lhs_.shape();
    let rhs_shape = rhs_.shape();
    let (m, k, n) = (
        lhs_shape[rank - 2],
        lhs_shape[rank - 1],
        rhs_shape[rank - 1],
    );

    // common parameters for gemm
    let (rsa, csa) = (lhs_strides[rank - 2], lhs_strides[rank - 1]);
    let (rsb, csb) = (rhs_strides[rank - 2], rhs_strides[rank - 1]);
    let (rsc, csc) = {
        let strides = c_.strides();
        (strides[rank - 2], strides[rank - 1])
    };
    let num_batches: usize = lhs_shape[..rank - 2].iter().product();
    let lhs_batch_size = lhs_.len() / num_batches;
    let rhs_batch_size = rhs_.len() / num_batches;
    let c_batch_size = c_.len() / num_batches;
    let ap_init = lhs.as_ptr();
    let bp_init = rhs.as_ptr();
    let cp_init = c.as_mut_ptr();

    use rayon::prelude::*;
    use std::slice;

    unsafe {
        let lhs_slice = slice::from_raw_parts(ap_init, lhs.len());
        let rhs_slice = slice::from_raw_parts(bp_init, rhs.len());
        let c_slice = slice::from_raw_parts_mut(cp_init, c.len());

        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<F, $ty>() {
                    let lhs_iter = lhs_slice.par_iter().step_by(lhs_batch_size);
                    let rhs_iter = rhs_slice.par_iter().step_by(rhs_batch_size);
                    let c_iter = c_slice.par_iter_mut().step_by(c_batch_size);

                    lhs_iter
                        .zip_eq(rhs_iter)
                        .zip_eq(c_iter)
                        .for_each(|((lhs, rhs), c)| {
                            ::matrixmultiply::$f(
                                m,
                                k,
                                n,
                                cast_as(&alpha),
                                lhs as *const F as *const _,
                                rsa,
                                csa,
                                rhs as *const F as *const _,
                                rsb,
                                csb,
                                cast_as(&beta),
                                c as *mut F as *mut _,
                                rsc,
                                csc,
                            );
                        });
                }
            };
        }
        kernel_call_def!(f32, sgemm);
        kernel_call_def!(f64, dgemm);
    }
}

#[inline]
fn batch_mat_mul_requires_copy(stride: &[ndarray::Ixs]) -> bool {
    let rank = stride.len();
    // unwrap is ok since stride.len() > 2
    let min_str = *stride[0..rank - 2].iter().min().unwrap();
    let row_str = stride[rank - 2];
    let col_str = stride[rank - 1];
    min_str < row_str || min_str < col_str
}

fn dot_shape_error(m: usize, k: usize, k2: usize, n: usize) -> String {
    match m.checked_mul(n) {
        Some(len) if len <= ::std::isize::MAX as usize => {}
        _ => {
            return format!("ndarray: shape {} × {} overflows isize", m, n);
        }
    }
    format!(
        "ndarray: inputs {} × {} and {} × {} are not compatible for matrix multiplication",
        m, k, k2, n
    )
}

// ========= Op impls =========

#[cfg(feature = "blas")]
use cblas_sys::CBLAS_LAYOUT::CblasRowMajor;
#[cfg(feature = "blas")]
use cblas_sys::CBLAS_TRANSPOSE::{CblasNoTrans, CblasTrans};
use ndarray::ShapeBuilder;

pub struct MatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct BatchMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl<T: Float> op::Op<T> for MatMul {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let mut a = ctx
            .input(0)
            .into_dimensionality::<ndarray::Ix2>()
            .expect("lhs input for MatMul must be 2D");
        let mut b = ctx
            .input(1)
            .into_dimensionality::<ndarray::Ix2>()
            .expect("rhs input for MatMul must be 2D");
        if self.transpose_a {
            a.swap_axes(0, 1);
        }
        if self.transpose_b {
            b.swap_axes(0, 1);
        }
        let ((m, k), (k2, n)) = (a.dim(), b.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            return Err(op::OpError::IncompatibleShape(dot_shape_error(m, k, k2, n)));
        }

        let lhs_s0 = a.strides()[0];
        let rhs_s0 = b.strides()[0];
        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        // A is Copy so this is safe
        let mut v = Vec::with_capacity(m * n);
        let mut c;
        unsafe {
            v.set_len(m * n);
            c = ndarray::Array::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }

        #[cfg(feature = "blas")]
        {
            mat_mul_impl_blas(T::one(), &a, &b, T::zero(), &mut c.view_mut());
        }
        #[cfg(not(feature = "blas"))]
        {
            mat_mul_impl_slow(T::one(), &a, &b, T::zero(), &mut c.view_mut());
        }
        ctx.append_output(c.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let opa = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(&ctx.input(1), false)
            .build(MatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder(ctx.graph())
            .append_input(&ctx.input(0), false)
            .append_input(gy, false)
            .build(MatMul {
                transpose_a: true,
                transpose_b: false,
            });

        ctx.append_input_grad(Some(opa));
        ctx.append_input_grad(Some(opb));
    }
}

impl<T: Float> op::Op<T> for BatchMatMul {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let mut x0 = ctx.input(0);
        let mut x1 = ctx.input(1);
        let rank0 = x0.ndim();
        let rank1 = x1.ndim();

        if rank0 < 2 {
            return Err(op::OpError::IncompatibleShape(format!(
                "BatchMatMul: Left-hand-side input's ndim must be >= 2, actual: {}",
                rank0
            )));
        }
        if rank1 < 2 {
            return Err(op::OpError::IncompatibleShape(format!(
                "BatchMatMul: Right-hand-side input's ndim must be >= 2, actual: {}",
                rank1
            )));
        }

        if self.transpose_a {
            x0.swap_axes(rank0 - 2, rank0 - 1);
        }

        if self.transpose_b {
            x1.swap_axes(rank1 - 2, rank1 - 1);
        }

        let shape0 = x0.shape();
        let shape1 = x1.shape();
        if rank0 != rank1 || shape0[..rank0 - 2] != shape1[..rank0 - 2] {
            return Err(op::OpError::IncompatibleShape(format!(
                "Input shapes mismatch: {:?} vs {:?}",
                shape0, shape1
            )));
        }

        let ret_shape = {
            let mut ret = shape0.to_vec();
            ret[rank0 - 2] = shape0[rank0 - 2];
            ret[rank0 - 1] = shape1[rank0 - 1];
            ret
        };
        // A is Copy so this is safe
        let size: usize = ret_shape.iter().product();
        let mut v = Vec::with_capacity(size);
        let mut c;
        unsafe {
            v.set_len(size);
            // BatchMatMul's ret val is a c-order array.
            c = ndarray::Array::from_shape_vec_unchecked(ret_shape, v);
        }
        #[cfg(feature = "blas")]
        {
            batch_mat_mul_impl_fast(T::one(), &x0, &x1, T::zero(), &mut c.view_mut());
        }
        #[cfg(not(feature = "blas"))]
        {
            batch_mat_mul_impl_slow(T::one(), &x0, &x1, T::zero(), &mut c.view_mut())
        }

        // reshape to dst shape with safe unwrapping
        ctx.append_output(c);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let opa = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(&ctx.input(1), false)
            .build(BatchMatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder(ctx.graph())
            .append_input(&ctx.input(0), false)
            .append_input(gy, false)
            .build(BatchMatMul {
                transpose_a: true,
                transpose_b: false,
            });

        ctx.append_input_grad(Some(opa));
        ctx.append_input_grad(Some(opb));
    }
}

pub struct TensordotPreprocess;

#[inline]
fn tensordot_preprocess<T: Float>(
    shape: &[usize],
    axes: &[usize],
    flip: bool,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let free = (0..shape.len())
        .filter(|i| !axes.contains(i))
        .collect::<Vec<usize>>();
    let mut free_dims = Vec::with_capacity(free.len());
    let mut prod_free_dims = 1;
    {
        for &i in &free {
            prod_free_dims *= shape[i];
            free_dims.push(T::from(shape[i]).unwrap());
        }
    }
    let prod_axes_dims = axes.iter().map(|&i| shape[i]).product::<usize>();

    // make perm
    let first = if flip { axes } else { &free };
    let second = if flip { &free } else { axes };
    let mut perm = Vec::with_capacity(first.len() + second.len());
    for &a in first {
        perm.push(T::from(a).unwrap());
    }
    for &a in second {
        perm.push(T::from(a).unwrap());
    }

    // make new shape
    let new_shape = if flip {
        vec![
            T::from(prod_axes_dims).unwrap(),
            T::from(prod_free_dims).unwrap(),
        ]
    } else {
        vec![
            T::from(prod_free_dims).unwrap(),
            T::from(prod_axes_dims).unwrap(),
        ]
    };

    (perm, new_shape, free_dims)
}

impl<T: Float> op::Op<T> for TensordotPreprocess {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x0 = ctx.input(0);
        let x1 = &ctx.input(1);
        let axes0 = crate::ndarray_ext::normalize_negative_axes(&ctx.input(2), x0.ndim());
        let axes1 = crate::ndarray_ext::normalize_negative_axes(&ctx.input(3), x1.ndim());

        let (perm0, new_shape0, mut free_dims0) = tensordot_preprocess(x0.shape(), &axes0, false);
        let (perm1, new_shape1, free_dims1) = tensordot_preprocess(x1.shape(), &axes1, true);
        free_dims0.extend(free_dims1);

        let r0 = NdArray::from_shape_vec(ndarray::IxDyn(&[free_dims0.len()]), free_dims0).unwrap();
        let r1 = NdArray::from_shape_vec(ndarray::IxDyn(&[perm0.len()]), perm0).unwrap();
        let r2 = NdArray::from_shape_vec(ndarray::IxDyn(&[perm1.len()]), perm1).unwrap();
        let r3 = NdArray::from_shape_vec(ndarray::IxDyn(&[new_shape0.len()]), new_shape0).unwrap();
        let r4 = NdArray::from_shape_vec(ndarray::IxDyn(&[new_shape1.len()]), new_shape1).unwrap();

        ctx.append_output(r0);
        ctx.append_output(r1);
        ctx.append_output(r2);
        ctx.append_output(r3);
        ctx.append_output(r4);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
