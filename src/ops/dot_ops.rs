/// Some gemm kernel usages are ported from ndarray
use crate::ndarray_ext::NdArray;
use crate::{op, NdArrayViewMut};
#[cfg(feature = "mkl")]
use crate::same_type;
use crate::tensor::Tensor;
use crate::NdArrayView;
use crate::Float;
use ndarray;
use ndarray::{ArrayView2, ArrayViewMut2};
#[cfg(feature = "mkl")]
use ndarray::Dimension;
#[cfg(feature = "mkl")]
use std::cmp;
#[cfg(feature = "mkl")]
use std::mem;
#[cfg(feature = "mkl")]
use crate::ops::mkl_ffi::*;
#[cfg(feature = "mkl")]
use crate::ndarray_ext::{get_batch_ptrs, get_batch_ptrs_mut};


#[cfg(feature = "mkl")]
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

#[cfg(feature = "mkl")]
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
    is_blas_nd(a.shape(), strides[rank - 2], strides[rank - 1], MemoryOrder::C)
}

#[cfg(feature = "mkl")]
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

#[cfg(feature = "mkl")]
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
    is_blas_nd(a.shape(), strides[rank - 2], strides[rank - 1], MemoryOrder::C)
}

#[cfg(feature = "mkl")]
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
    if (stride0 > MklInt::max_value() as isize || stride0 < MklInt::min_value() as isize)
        || (stride1 > MklInt::max_value() as isize || stride1 < MklInt::min_value() as isize)
    {
        return false;
    }
    if m > MklInt::max_value() as usize || n > MklInt::max_value() as usize {
        return false;
    }
    true
}

#[cfg(feature = "mkl")]
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
    if (s0 > MklInt::max_value() as isize || s0 < MklInt::min_value() as isize)
        || (s1 > MklInt::max_value() as isize || s1 < MklInt::min_value() as isize)
    {
        return false;
    }
    if m > MklInt::max_value() as usize || n > MklInt::max_value() as usize {
        return false;
    }
    true
}


// mkl version of ndarray's mat_mul_impl
#[cfg(feature = "mkl")]
fn mat_mul_impl_blas<F: Float>(
    alpha: F,
    lhs: &ArrayView2<'_, F>,
    rhs: &ArrayView2<'_, F>,
    beta: F,
    c: &mut ArrayViewMut2<'_, F>,
)
{
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
        let mut lhs_trans = CblasTranspose::CblasNoTrans;
        let mut rhs_trans = CblasTranspose::CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
            mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_ = lhs_.reversed_axes();
            lhs_trans = CblasTranspose::CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_ = rhs_.reversed_axes();
            rhs_trans = CblasTranspose::CblasTrans;
        }

        macro_rules! call_kernel_def {
            ($ty:ty, $f:ident) => {
                if blas_row_major_2d::<$ty, _>(&lhs_)
                    && blas_row_major_2d::<$ty, _>(&rhs_)
                    && blas_row_major_2d_mut::<$ty, _>(&c_)
                {
                    let (m, k) = match lhs_trans {
                        CblasTranspose::CblasNoTrans => lhs_.dim(),
                        _ => {
                            let (rows, cols) = lhs_.dim();
                            (cols, rows)
                        }
                    };
                    let n = match rhs_trans {
                        CblasTranspose::CblasNoTrans => rhs_.raw_dim()[1],
                        _ => rhs_.raw_dim()[0],
                    };
                    // adjust strides, these may [1, 1] for column matrices
                    let lhs_stride = cmp::max(lhs_.strides()[0] as MklInt, k as MklInt);
                    let rhs_stride = cmp::max(rhs_.strides()[0] as MklInt, n as MklInt);
                    let c_stride = cmp::max(c_.strides()[0] as MklInt, n as MklInt);

                    // gemm is C ← αA^Op B^Op + βC
                    // Where Op is notrans/trans/conjtrans
                    unsafe {
                        $f(
                            CBLAS_ROW_MAJOR,
                            lhs_trans,
                            rhs_trans,
                            m as MklInt,               // m, rows of Op(a)
                            n as MklInt,               // n, cols of Op(b)
                            k as MklInt,               // k, cols of Op(a)
                            crate::cast_as(&alpha),               // alpha
                            lhs_.as_ptr() as *const _, // a
                            lhs_stride,                    // lda
                            rhs_.as_ptr() as *const _, // b
                            rhs_stride,                    // ldb
                            crate::cast_as(&beta),                // beta
                            c_.as_mut_ptr() as *mut _,     // c
                            c_stride,                      // ldc
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
#[cfg(feature = "mkl")]
fn batch_mat_mul_impl<F: Float>(
    alpha: F,
    lhs: &NdArrayView<'_, F>,
    rhs: &NdArrayView<'_, F>,
    beta: F,
    c: &mut NdArrayViewMut<'_, F>,
)
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let rank = lhs.ndim();
    let (mut m, a, mut n) = (lhs_shape[rank - 2], lhs_shape[rank - 1], rhs_shape[rank - 1]);

    {
        // Use `c` for c-order and `f` for an f-order matrix
        // We can handle c * c, f * f generally and
        // c * f and f * c if the `f` matrix is square.
        let mut lhs_ = lhs.view();
        let mut rhs_ = rhs.view();
        let mut c_ = c.view_mut();
        let mut lhs_strides = lhs_.strides();
        let mut rhs_strides = rhs_.strides();

        // copy if batch dims appear in last two dims.
        let mut copied_lhs = None;
        let mut copied_rhs = None;
        if batch_mat_mul_requires_copy(lhs_strides) {
            copied_lhs = Some(crate::ndarray_ext::deep_copy(&lhs_));
            lhs_ = copied_lhs.as_ref().unwrap().view();
            lhs_strides = lhs_.strides();
        }
        if batch_mat_mul_requires_copy(rhs_strides) {
            copied_rhs = Some(crate::ndarray_ext::deep_copy(&rhs_));
            rhs_ = copied_rhs.as_ref().unwrap().view();
            rhs_strides = rhs_.strides();
        }

        let lhs_s0 = lhs_strides[rank - 2];
        let rhs_s0 = rhs_strides[rank - 2];
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;

        let mut lhs_trans = CblasTranspose::CblasNoTrans;
        let mut rhs_trans = CblasTranspose::CblasNoTrans;

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
            lhs_trans = CblasTranspose::CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_.swap_axes(rank - 2, rank - 1);
            rhs_trans = CblasTranspose::CblasTrans;
        }
        let batch_size: usize = lhs_shape[..rank - 2].iter().product();

        macro_rules! call_kernel_def {
            ($ty:ty, $f:ident) => {
                if blas_row_major_nd::<$ty, _>(&lhs_)
                    && blas_row_major_nd::<$ty, _>(&rhs_)
                    && blas_row_major_nd_mut::<$ty, _>(&c_)
                {
                    let (m, k) = match lhs_trans {
                        CblasTranspose::CblasNoTrans => {
                            let s = lhs_.shape();
                            (s[rank - 2], s[rank - 1])
                        },
                        _ => {
                            let s = lhs_.shape();
                            (s[rank - 1], s[rank - 2])
                        }
                    };
                    let n = match rhs_trans {
                        CblasTranspose::CblasNoTrans => rhs_.raw_dim()[rank - 1],
                        _ => rhs_.raw_dim()[rank - 2],
                    };
                    // adjust strides, these may [1, 1] for column matrices
                    let lhs_stride = cmp::max(lhs_.strides()[rank - 2] as MklInt, k as MklInt);
                    let rhs_stride = cmp::max(rhs_.strides()[rank - 2] as MklInt, n as MklInt);
                    let c_stride = cmp::max(c_.strides()[rank - 2] as MklInt, n as MklInt);

                    unsafe {
                        const GROUP_COUNT: usize = 1;  // Fixed
                        $f(
                            CBLAS_ROW_MAJOR,
                            [lhs_trans; GROUP_COUNT].as_ptr(),
                            [rhs_trans; GROUP_COUNT].as_ptr(),
                            [m as MklInt; GROUP_COUNT].as_ptr(),
                            [n as MklInt; GROUP_COUNT].as_ptr(),
                            [k as MklInt; GROUP_COUNT].as_ptr(),
                            [crate::cast_as(&alpha); GROUP_COUNT].as_ptr(),             // alpha
                            get_batch_ptrs(batch_size, lhs_.as_ptr(), lhs_.len()).as_ptr(), // a array
                            [lhs_stride; GROUP_COUNT].as_ptr(),
                            get_batch_ptrs(batch_size, rhs_.as_ptr(), rhs_.len()).as_ptr(), // b array
                            [rhs_stride; GROUP_COUNT].as_ptr(),
                            [crate::cast_as(&beta); GROUP_COUNT].as_ptr(),               // alpha
                            get_batch_ptrs_mut(batch_size, c_.as_mut_ptr(), c_.len()).as_mut_ptr(), // c array
                            [c_stride; GROUP_COUNT].as_ptr(),
                            GROUP_COUNT as MklInt,
                            [batch_size as MklInt; GROUP_COUNT].as_ptr()
                        );
                    }
                    return;
                }
            };
        }
        call_kernel_def!(f32, cblas_sgemm_batch);
        call_kernel_def!(f64, cblas_dgemm_batch);
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
) where
{
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
                        crate::cast_as(&alpha),
                        ap as *const _,
                        lhs.strides()[0],
                        lhs.strides()[1],
                        bp as *const _,
                        rhs.strides()[0],
                        rhs.strides()[1],
                        crate::cast_as(&beta),
                        cp as *mut _,
                        rsc,
                        csc,
                    );
                }
            }
        }
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
) where
{
    let mut lhs_ = lhs.view();
    let mut rhs_ = rhs.view();
    let c_ = c.view_mut();
    let mut lhs_strides = lhs_.strides();
    let mut rhs_strides = rhs_.strides();
    let rank = lhs_strides.len();
    let lhs_requires_copy = batch_mat_mul_requires_copy(lhs_strides);
    let rhs_requires_copy = batch_mat_mul_requires_copy(rhs_strides);

    let mut copied_lhs = None;
    let mut copied_rhs = None;
    // Update lhs, rhs info with copied ones
    {
        if lhs_requires_copy {
            copied_lhs = Some(crate::ndarray_ext::deep_copy(&lhs_));
            lhs_ = copied_lhs.as_ref().unwrap().view();
            lhs_strides = lhs_.strides();
        }
        if rhs_requires_copy {
            copied_rhs = Some(crate::ndarray_ext::deep_copy(&rhs_));
            rhs_ = copied_rhs.as_ref().unwrap().view();
            rhs_strides = rhs_.strides();
        }
    }

    let lhs_shape = lhs_.shape();
    let rhs_shape = rhs_.shape();
    let (m, k, n) = (lhs_shape[rank - 2], lhs_shape[rank - 1], rhs_shape[rank - 1]);

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
    unsafe {
        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<F, $ty>() {
                    for batch_i in 0..num_batches {
                        let a_pos = (lhs_batch_size * batch_i) as isize;
                        let b_pos = (rhs_batch_size * batch_i) as isize;
                        let c_pos = (c_batch_size * batch_i) as isize;
                        let ap = ap_init.offset(a_pos);
                        let bp = bp_init.offset(b_pos);
                        let cp = cp_init.offset(c_pos);
                        ::matrixmultiply::$f(
                            m,
                            k,
                            n,
                            crate::cast_as(&alpha),
                            ap as *const _,
                            rsa,
                            csa,
                            bp as *const _,
                            rsb,
                            csb,
                            crate::cast_as(&beta),
                            cp as *mut _,
                            rsc,
                            csc,
                        );
                    }
                }
            }
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

#[cold]
#[inline(never)]
fn dot_shape_error(m: usize, k: usize, k2: usize, n: usize) -> ! {
    match m.checked_mul(n) {
        Some(len) if len <= ::std::isize::MAX as usize => {}
        _ => panic!("ndarray: shape {} × {} overflows isize", m, n),
    }
    panic!(
        "ndarray: inputs {} × {} and {} × {} are not compatible for matrix multiplication",
        m, k, k2, n
    );
}


// ========= Op impls =========

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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let mut a = ctx.input(0).into_dimensionality::<ndarray::Ix2>().expect("lhs input for MatMul must be 2D");
        let mut b = ctx.input(1).into_dimensionality::<ndarray::Ix2>().expect("rhs input for MatMul must be 2D");
        if self.transpose_a {
            a.swap_axes(0, 1);
        }
        if self.transpose_b {
            b.swap_axes(0, 1);
        }
        let ((m, k), (k2, n)) = (a.dim(), b.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            dot_shape_error(m, k, k2, n);
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

        #[cfg(feature = "mkl")] {
            mat_mul_impl_blas(T::one(), &a, &b, T::zero(), &mut c.view_mut());
        }
        #[cfg(not(feature = "mkl"))] {
            mat_mul_impl_slow(T::one(), &a, &b, T::zero(), &mut c.view_mut());
        }
        ctx.append_output(Ok(crate::ArrRepr::Owned(c.into_dyn())));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let gy = &ctx.output_grad();
        let opa = Tensor::builder().set_inputs(&[gy, &ctx.input(1)]).build(
            s,
            MatMul {transpose_a: false, transpose_b: true},
        );

        let opb = Tensor::builder().set_inputs(&[&ctx.input(0), gy]).build(
            s,
            MatMul {transpose_a: true, transpose_b: false},
        );

        ctx.set_input_grads(vec![Some(opa), Some(opb)]);
    }
}

impl<T: Float> op::Op<T> for BatchMatMul {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let mut x0 = ctx.input(0);
        let mut x1 = ctx.input(1);
        let rank0 = x0.ndim();
        let rank1 = x1.ndim();

        assert!(
            rank0 >= 2,
            "BatchMatMul: Left-hand-side input's ndim must be >= 2, actual: {}",
            rank0
        );
        assert!(
            rank1 >= 2,
            "BatchMatMul: Right-hand-side input's ndim must be >= 2, actual: {}",
            rank1
        );

        if self.transpose_a {
            x0.swap_axes(rank0 - 2, rank0 - 1);
        }

        if self.transpose_b {
            x1.swap_axes(rank1 - 2, rank1 - 1);
        }

        let shape0 = x0.shape();
        let shape1 = x1.shape();
        if rank0 != rank1 || shape0[..rank0 - 2] != shape1[..rank0 - 2] {
            panic!("Input shapes mismatch: {:?} vs {:?}", shape0, shape1);
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
        #[cfg(feature = "mkl")] {
            batch_mat_mul_impl(T::one(), &x0, &x1, T::zero(), &mut c.view_mut());
        }
        #[cfg(not(feature = "mkl"))] {
            batch_mat_mul_impl_slow(T::one(), &x0, &x1, T::zero(), &mut c.view_mut())
        }

        // reshape to dst shape with safe unwrapping
        ctx.append_output(Ok(crate::ArrRepr::Owned(c)));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let opa = Tensor::builder().set_inputs(&[gy, &ctx.input(1)]).build(
            ctx.graph(),
            BatchMatMul {
                transpose_a: false,
                transpose_b: true,
            },
        );

        let opb = Tensor::builder().set_inputs(&[&ctx.input(0), gy]).build(
            ctx.graph(),
            BatchMatMul {
                transpose_a: true,
                transpose_b: false,
            },
        );

        ctx.set_input_grads(vec![Some(opa), Some(opb)]);
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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
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

        ctx.append_output(Ok(crate::ArrRepr::Owned(r0)));
        ctx.append_output(Ok(crate::ArrRepr::Owned(r1)));
        ctx.append_output(Ok(crate::ArrRepr::Owned(r2)));
        ctx.append_output(Ok(crate::ArrRepr::Owned(r3)));
        ctx.append_output(Ok(crate::ArrRepr::Owned(r4)));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None; 4]);
    }
}
