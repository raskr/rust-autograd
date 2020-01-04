// Disable lint caused by a workaround in ndarray caused by a known issue in the Rust compiler
// see also: https://github.com/rust-ndarray/ndarray/issues/474
// and https://github.com/rust-lang/rust/issues/23014
#![allow(clippy::deref_addrof)]

use crate::ndarray_ext::NdArray;
use crate::op;
#[cfg(feature = "mkl")]
use crate::same_type;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;
#[cfg(feature = "mkl")]
use std::mem;

#[cfg(feature = "mkl")]
type MklInt = i64;

#[cfg(feature = "mkl")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    // CblasConjTrans = 113,
}

#[cfg(feature = "mkl")]
type CblasLayout = usize;

#[cfg(feature = "mkl")]
extern "C" {
    // sgemm from intel MKL
    fn cblas_sgemm(
        layout: CblasLayout,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: libc::c_float,
        a: *const libc::c_float,
        lda: MklInt,
        b: *const libc::c_float,
        ldb: MklInt,
        beta: libc::c_float,
        c: *mut libc::c_float,
        ldc: MklInt,
    );

    // dgemm from intel MKL
    fn cblas_dgemm(
        layout: CblasLayout,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: libc::c_double,
        a: *const libc::c_double,
        lda: MklInt,
        b: *const libc::c_double,
        ldb: MklInt,
        beta: libc::c_double,
        c: *mut libc::c_double,
        ldc: MklInt,
    );

    // Batched sgemm from intel MKL
    fn cblas_sgemm_batch(
        layout: CblasLayout,
        transa_array: *const CblasTranspose, // batch of CblasTranspose
        transb_array: *const CblasTranspose, // batch of CblasTranspose
        m_array: *const MklInt,              // batch of m
        n_array: *const MklInt,              // batch of n
        k_array: *const MklInt,              // batch of k
        alpha_array: *const libc::c_float,   // batch of alpha
        a_array: *const *const libc::c_float, // a
        lda_array: *const MklInt,            // batch of lda
        b_array: *const *const libc::c_float, // b
        ldb_array: *const MklInt,            // batch of ldb
        beta_array: *const libc::c_float,    // batch of beta
        c_array: *mut *mut libc::c_float,    // c
        ldc_array: *const MklInt,            // batch of odc
        group_count: MklInt,                 // batch size
        group_size: *const MklInt,
    ); // num of matrices in each batch
       // Batched sgemm from intel MKL

    fn cblas_dgemm_batch(
        layout: CblasLayout,
        transa_array: *const CblasTranspose, // batch of CblasTranspose
        transb_array: *const CblasTranspose, // batch of CblasTranspose
        m_array: *const MklInt,              // batch of m
        n_array: *const MklInt,              // batch of n
        k_array: *const MklInt,              // batch of k
        alpha_array: *const libc::c_double,  // batch of alpha
        a_array: *const *const libc::c_double, // a
        lda_array: *const MklInt,            // batch of lda
        b_array: *const *const libc::c_double, // b
        ldb_array: *const MklInt,            // batch of ldb
        beta_array: *const libc::c_double,   // batch of beta
        c_array: *mut *mut libc::c_double,   // c
        ldc_array: *const MklInt,            // batch of odc
        group_count: MklInt,                 // batch size
        group_size: *const MklInt,
    ); // num of matrices in each batch
}

#[cfg(feature = "mkl")]
#[inline]
pub fn cblas_sgemm_wrapper(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    let lda = if trans_a { m } else { k } as MklInt;
    let ldb = if trans_b { k } else { n } as MklInt;
    let ldc = n as MklInt;
    let trans_a = if trans_a {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    let trans_b = if trans_b {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    unsafe {
        const CBLAS_ROW_MAGER: usize = 101;
        cblas_sgemm(
            CBLAS_ROW_MAGER,
            trans_a,
            trans_b,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

#[test]
#[cfg(feature = "mkl")]
fn test_sgemm() {
    let x = vec![1., 2., 3., 4.]; // (2, 2)
    let y = vec![1., 2., 3., 4.]; // (2, 2)
    unsafe {
        let mut z = crate::uninitialized_vec::<f32>(4); // (2, 2, 2)

        cblas_sgemm_wrapper(
            false,
            false,
            2,  // m
            2,  // n
            2,  // k
            1., // alpha
            x.as_ptr(),
            y.as_ptr(), // b
            0.,         // beta
            z.as_mut_ptr(),
        );
        assert_eq!(z, vec![7., 10., 15., 22.]);
    }
}

#[inline]
#[cfg(feature = "mkl")]
pub fn cblas_dgemm_wrapper(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
) {
    let lda = if trans_a { m } else { k } as MklInt;
    let ldb = if trans_b { k } else { n } as MklInt;
    let ldc = n as MklInt;
    let trans_a = if trans_a {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    let trans_b = if trans_b {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    unsafe {
        const CBLAS_ROW_MAGER: usize = 101;
        cblas_dgemm(
            CBLAS_ROW_MAGER,
            trans_a,
            trans_b,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
}

#[inline]
#[cfg(feature = "mkl")]
pub fn cblas_sgemm_batch_wrapper(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: &[f32],
    a_array: Vec<*const f32>,
    b_array: Vec<*const f32>,
    beta: &[f32],
    c_array: Vec<*const f32>,
    group_count: usize,
    size_per_group: usize,
) {
    let size_per_group = size_per_group as usize;
    let lda = if trans_a { m } else { k } as MklInt;
    let ldb = if trans_b { k } else { n } as MklInt;
    let ldc = n as MklInt;
    let trans_a = if trans_a {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    let trans_b = if trans_b {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    unsafe {
        const CBLAS_ROW_MAGER: usize = 101;
        cblas_sgemm_batch(
            CBLAS_ROW_MAGER,
            vec![trans_a; group_count].as_slice().as_ptr(),
            vec![trans_b; group_count].as_slice().as_ptr(),
            vec![m as MklInt; group_count].as_slice().as_ptr(),
            vec![n as MklInt; group_count].as_slice().as_ptr(),
            vec![k as MklInt; group_count].as_slice().as_ptr(),
            alpha.as_ptr(),
            mem::transmute(a_array.as_slice().as_ptr()), // safe
            vec![lda as MklInt; group_count].as_slice().as_ptr(),
            mem::transmute(b_array.as_slice().as_ptr()), // safe
            vec![ldb as MklInt; group_count].as_slice().as_ptr(),
            beta.as_ptr(),
            mem::transmute(c_array.as_slice().as_ptr()), // ???
            vec![ldc as MklInt; group_count].as_slice().as_ptr(),
            group_count as MklInt,
            vec![size_per_group as MklInt; group_count]
                .as_slice()
                .as_ptr(),
        );
    }
}

#[inline]
#[cfg(feature = "mkl")]
pub fn cblas_dgemm_batch_wrapper(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: &[f64],
    a_array: Vec<*const f64>,
    b_array: Vec<*const f64>,
    beta: &[f64],
    c_array: Vec<*const f64>,
    group_count: usize,
    size_per_group: usize,
) {
    let size_per_group = size_per_group as usize;
    let lda = if trans_a { m } else { k } as MklInt;
    let ldb = if trans_b { k } else { n } as MklInt;
    let ldc = n as MklInt;
    let trans_a = if trans_a {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    let trans_b = if trans_b {
        CblasTranspose::CblasTrans
    } else {
        CblasTranspose::CblasNoTrans
    };
    unsafe {
        const CBLAS_ROW_MAGER: usize = 101;
        cblas_dgemm_batch(
            CBLAS_ROW_MAGER,
            vec![trans_a; group_count].as_slice().as_ptr(),
            vec![trans_b; group_count].as_slice().as_ptr(),
            vec![m as MklInt; group_count].as_slice().as_ptr(),
            vec![n as MklInt; group_count].as_slice().as_ptr(),
            vec![k as MklInt; group_count].as_slice().as_ptr(),
            alpha.as_ptr(),
            mem::transmute(a_array.as_slice().as_ptr()), // safe
            vec![lda as MklInt; group_count].as_slice().as_ptr(),
            mem::transmute(b_array.as_slice().as_ptr()), // safe
            vec![ldb as MklInt; group_count].as_slice().as_ptr(),
            beta.as_ptr(),
            mem::transmute(c_array.as_slice().as_ptr()), // ???
            vec![ldc as MklInt; group_count].as_slice().as_ptr(),
            group_count as MklInt,
            vec![size_per_group as MklInt; group_count]
                .as_slice()
                .as_ptr(),
        );
    }
}

#[test]
#[cfg(feature = "mkl")]
fn test_dgemm_batch_trans_a() {
    let batch = 2;
    let w = vec![0., 1., 2., 3., 4., 5.]; // (2, 3)
    let x = vec![0., 1., 2., 3., 4., 5., 6., 7.]; // (2, 2, 2)
    unsafe {
        let z = crate::uninitialized_vec::<f64>(12); // (2, 2, 2)
        let m = 3; // row of op(a)
        let n = 2; // col of op(b)
        let k = 2; // col of op(a)
        cblas_dgemm_batch_wrapper(
            true,
            false,
            m,
            n,
            k,
            &[1.],              // alpha
            vec![&w[0], &w[0]], // a
            get_region_heads(batch, x.as_ptr(), x.len()),
            &[0.], // beta
            get_region_heads(batch, z.as_ptr(), z.len()),
            1,
            batch,
        );
        assert_eq!(
            z,
            vec![6., 9., 8., 13., 10., 17., 18., 21., 28., 33., 38., 45.]
        );
    }
}

#[test]
#[cfg(feature = "mkl")]
fn test_dgemm_batch() {
    let batch = 2;
    let x = vec![0., 1., 2., 3.]; // (2, 2)
    let y = vec![0., 1., 2., 3., 4., 5., 6., 7.]; // (2, 2, 2)
    unsafe {
        let z = crate::uninitialized_vec::<f64>(8); // (2, 2, 2)

        cblas_dgemm_batch_wrapper(
            false,
            false,
            2,                                            // m
            2,                                            // n
            2,                                            // k
            &[1.],                                        // alpha
            vec![&x[0], &x[0]],                           // a
            get_region_heads(batch, y.as_ptr(), y.len()), // b
            &[0.],                                        // beta
            get_region_heads(batch, z.as_ptr(), z.len()), // c
            1,
            batch,
        );
        assert_eq!(z, vec![2., 3., 6., 11., 6., 7., 26., 31.]);
    }
}

// `Tensordot` is implemented in `ops/mod.rs`.

pub struct MatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct BatchMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

#[cfg(feature = "mkl")]
macro_rules! mkl_mm {
    ($f:expr, $x0:expr, $x1:expr, $x0_shape:expr, $x1_shape:expr, $self:expr, $typ:ty) => {{
        let s0l = $x0.strides()[0];
        let s0r = $x1.strides()[0];
        let row0 = $x0_shape[0]; // rows of a
        let col0 = $x0_shape[1]; // cols of a
        let row1 = $x1_shape[0]; // rows of b
        let col1 = $x1_shape[1]; // cols of b
        let transpose_a = if s0l == 1 && row0 == 1 {
            !$self.transpose_a
        } else {
            $self.transpose_a
        };
        let transpose_b = if s0r == 1 && row1 == 1 {
            !$self.transpose_b
        } else {
            $self.transpose_b
        };
        let m = if transpose_a { col0 } else { row0 };
        let n = if transpose_b { row1 } else { col1 };
        let k = if transpose_a { row0 } else { col0 };
        let ret_row = if $self.transpose_a { col0 } else { row0 };
        let ret_col = if $self.transpose_b { row1 } else { col1 };

        unsafe {
            let mut c = crate::uninitialized_vec::<T>(ret_row * ret_col);
            $f(
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                1.,
                $x0.as_ptr() as *const $typ,
                $x1.as_ptr() as *const $typ,
                0.,
                c.as_mut_ptr() as *mut $typ,
            );
            vec![Ok(crate::ArrRepr::Owned(
                NdArray::from_shape_vec(ndarray::IxDyn(&[ret_row, ret_col]), c).unwrap(),
            ))]
        }
    }};
}

#[cfg(feature = "mkl")]
enum BatchMatMulAxesInfo {
    Trans,
    NoTrans,
    Dirty,
}

#[inline]
#[cfg(feature = "mkl")]
fn batch_mm_axes_info(stride: &[ndarray::Ixs]) -> BatchMatMulAxesInfo {
    let rank = stride.len();
    let slice = &stride[0..rank - 2];
    let c = *slice.iter().min().unwrap();
    let a = stride[rank - 2]; // row
    let b = stride[rank - 1]; // col
    if c > a && a > b {
        BatchMatMulAxesInfo::NoTrans
    } else if c > a && a < b {
        BatchMatMulAxesInfo::Trans
    } else {
        BatchMatMulAxesInfo::Dirty
    }
}

#[cfg(feature = "mkl")]
macro_rules! mkl_batch_mm {
    ($f:expr, $x0:expr, $x1:expr, $row0:expr,
        $col0:expr, $row1:expr, $col1:expr, $ret_shape:expr, $self:expr, $batch_size:expr) => {{
        let mut transpose_a = $self.transpose_a;
        let mut transpose_b = $self.transpose_b;

        let s0 = $x0.strides();
        let s1 = $x1.strides();
        let info0 = batch_mm_axes_info(s0);
        let info1 = batch_mm_axes_info(s1);

        let mut copied0 = None;
        let mut copied1 = None;

        match info0 {
            BatchMatMulAxesInfo::Trans => {
                transpose_a = !transpose_a;
            }
            BatchMatMulAxesInfo::Dirty => {
                copied0 = Some(crate::ndarray_ext::deep_copy($x0));
            }
            _ => {}
        }
        match info1 {
            BatchMatMulAxesInfo::Trans => {
                transpose_b = !transpose_b;
            }
            BatchMatMulAxesInfo::Dirty => {
                copied1 = Some(crate::ndarray_ext::deep_copy($x1));
            }
            _ => {}
        }
        let m = if transpose_a { $col0 } else { $row0 }; // rows of a
        let n = if transpose_b { $row1 } else { $col1 }; // cols of b
        let k = if transpose_a { $row0 } else { $col0 }; // cols of a

        unsafe {
            let ret = crate::uninitialized_vec($ret_shape.iter().product());

            $f(
                transpose_a,
                transpose_b,
                m,
                n,
                k,
                &[1.],
                get_region_heads(
                    $batch_size,
                    if let Some(v) = copied0 {
                        v.as_ptr()
                    } else {
                        $x0.as_ptr()
                    },
                    $x0.len(),
                ), // a array
                get_region_heads(
                    $batch_size,
                    if let Some(v) = copied1 {
                        v.as_ptr()
                    } else {
                        $x1.as_ptr()
                    },
                    $x1.len(),
                ), // b array
                &[0.],
                get_region_heads($batch_size, ret.as_ptr(), ret.len()), // c array
                1,
                $batch_size,
            );
            vec![Ok(crate::ArrRepr::Owned(
                NdArray::from_shape_vec(ndarray::IxDyn($ret_shape.as_slice()), ret).unwrap(),
            ))]
        }
    }};
}

impl<T: Float> op::Op<T> for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let x0_shape = x0.shape();
        let x1_shape = x1.shape();

        assert_eq!(
            x0_shape.len(),
            2,
            "First input to matmul should be a matrix"
        );
        assert_eq!(
            x1_shape.len(),
            2,
            "Second input to matmul should be a matrix"
        );
        #[cfg(feature = "mkl")]
        {
            if same_type::<T, f32>() {
                mkl_mm!(cblas_sgemm_wrapper, x0, x1, x0_shape, x1_shape, self, f32)
            } else if same_type::<T, f64>() {
                mkl_mm!(cblas_dgemm_wrapper, x0, x1, x0_shape, x1_shape, self, f64)
            } else {
                panic!("gemm supports only f32 and f64.")
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            // unwrap is always safe
            let mut a = x0.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            let mut b = x1.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            if self.transpose_a {
                a.swap_axes(0, 1);
            }
            if self.transpose_b {
                b.swap_axes(0, 1);
            }
            let ret = a.dot(&b).into_dyn();
            vec![Ok(crate::ArrRepr::Owned(ret))]
        }
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let opa = Tensor::builder()
            .set_inputs(vec![gy, inputs[1]])
            .build(MatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .build(MatMul {
                transpose_a: true,
                transpose_b: false,
            });

        vec![Some(opa), Some(opb)]
    }
}

#[inline]
pub fn get_region_heads<A: Float, B>(
    batch_size: usize,
    head: *const A,
    whole_size: usize,
) -> Vec<*const B> {
    let size_per_sample = whole_size / batch_size;
    let mut ret = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        unsafe {
            ret.push(head.add(i * size_per_sample) as *const B);
        }
    }
    ret
}

impl<T: Float> op::Op<T> for BatchMatMul {
    fn name(&self) -> &str {
        "BatchMatMul"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let shape0 = x0.shape();
        let shape1 = x1.shape();
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

        if rank0 != rank1 || shape0[..rank0 - 2] != shape1[..rank0 - 2] {
            panic!("Input shapes mismatch: {:?} vs {:?}", shape0, shape1);
        }

        let row0 = shape0[rank0 - 2];
        let col0 = shape0[rank0 - 1];
        let col1 = shape1[rank0 - 1];

        #[cfg(feature = "mkl")]
        {
            let batch_size: usize = shape0[..rank0 - 2].iter().product();
            let row1 = shape1[rank1 - 2];
            let ret_shape = {
                let mut ret = shape0.to_vec();
                ret[rank0 - 2] = if self.transpose_a { col0 } else { row0 };
                ret[rank0 - 1] = if self.transpose_b { row1 } else { col1 };
                ret
            };
            if same_type::<T, f32>() {
                mkl_batch_mm!(
                    cblas_sgemm_batch_wrapper,
                    x0,
                    x1,
                    row0,
                    col0,
                    row1,
                    col1,
                    ret_shape,
                    self,
                    batch_size
                )
            } else if same_type::<T, f64>() {
                mkl_batch_mm!(
                    cblas_dgemm_batch_wrapper,
                    x0,
                    x1,
                    row0,
                    col0,
                    row1,
                    col1,
                    ret_shape,
                    self,
                    batch_size
                )
            } else {
                panic!("gemm supports only f32 and f64.")
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            use rayon::iter::*;
            // squashes dims (remains last two dims)
            // unwrap is always safe
            let x0_flattened = {
                let mut a = x0
                    .view()
                    .into_shape((x0.len() / row0 / col0, row0, col0))
                    .unwrap();
                if self.transpose_a {
                    a.swap_axes(1, 2);
                }
                a
            };
            let row1 = shape1[rank0 - 2];
            let x1_flattened = {
                let mut b = x1
                    .view()
                    .into_shape((x1.len() / row1 / col1, row1, col1))
                    .unwrap();
                if self.transpose_b {
                    b.swap_axes(1, 2);
                }
                b
            };

            // parallel mm
            let dot = (0..x0_flattened.shape()[0] as isize)
                .into_par_iter()
                .map(|i| {
                    let x0_mat = x0_flattened
                        .slice(s![i..=i, .., ..])
                        .index_axis_move(ndarray::Axis(0), 0)
                        .to_owned();
                    let x1_mat = x1_flattened
                        .slice(s![i..=i, .., ..])
                        .index_axis_move(ndarray::Axis(0), 0)
                        .to_owned();
                    x0_mat.dot(&x1_mat).into_dyn()
                })
                .collect::<Vec<_>>();

            // owned to ref
            let mut dot_view = Vec::with_capacity(dot.len());
            for dot_i in &dot {
                // insert new dim
                let mut shape = dot_i.shape().to_vec();
                shape.insert(0, 1);
                dot_view.push(dot_i.view().into_shape(shape).unwrap());
            }

            // stack dot result
            let stacked = ndarray::stack(ndarray::Axis(0), dot_view.as_slice()).unwrap();

            let dst_shape = {
                let stacked_shape = stacked.shape();
                shape0[..rank0 - 2]
                    .iter()
                    .chain(&[stacked_shape[1], stacked_shape[2]])
                    .cloned()
                    .collect::<Vec<usize>>()
            };

            // reshape to dst shape with safe unwrapping
            vec![Ok(crate::ArrRepr::Owned(
                stacked
                    .into_shape(ndarray::IxDyn(dst_shape.as_slice()))
                    .unwrap(),
            ))]
        }
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let opa = Tensor::builder()
            .set_inputs(vec![gy, inputs[1]])
            .build(BatchMatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .build(BatchMatMul {
                transpose_a: true,
                transpose_b: false,
            });

        vec![Some(opa), Some(opb)]
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
    fn name(&self) -> &str {
        "TensordotPreprocess"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let axes0 = crate::ndarray_ext::normalize_negative_axes(&xs[2], x0.ndim());
        let axes1 = crate::ndarray_ext::normalize_negative_axes(&xs[3], x1.ndim());

        let (perm0, new_shape0, mut free_dims0) = tensordot_preprocess(x0.shape(), &axes0, false);
        let (perm1, new_shape1, free_dims1) = tensordot_preprocess(x1.shape(), &axes1, true);
        free_dims0.extend(free_dims1);

        let r0 = NdArray::from_shape_vec(ndarray::IxDyn(&[free_dims0.len()]), free_dims0).unwrap();
        let r1 = NdArray::from_shape_vec(ndarray::IxDyn(&[perm0.len()]), perm0).unwrap();
        let r2 = NdArray::from_shape_vec(ndarray::IxDyn(&[perm1.len()]), perm1).unwrap();
        let r3 = NdArray::from_shape_vec(ndarray::IxDyn(&[new_shape0.len()]), new_shape0).unwrap();
        let r4 = NdArray::from_shape_vec(ndarray::IxDyn(&[new_shape1.len()]), new_shape1).unwrap();

        vec![
            Ok(crate::ArrRepr::Owned(r0)),
            Ok(crate::ArrRepr::Owned(r1)),
            Ok(crate::ArrRepr::Owned(r2)),
            Ok(crate::ArrRepr::Owned(r3)),
            Ok(crate::ArrRepr::Owned(r4)),
        ]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None; 4]
    }
}
