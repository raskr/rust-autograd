#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
extern crate libc;
#[cfg(not(feature = "mkl"))]
extern crate matrixmultiply;
extern crate ndarray;
extern crate rayon;
use self::libc::{c_float, c_int};
#[allow(unused_imports)]
use self::rayon::iter::*;
use ndarray_ext::NdArray;
use std::f32;
use std::mem;
use std::slice;
use tensor::Tensor;

macro_rules! get_xw {
    ($op:expr, $yw:expr, $kw:expr) => {
        $op.stride * ($yw - 1) - 2 * $op.pad + ($op.dilation * ($kw - 1) + 1)
    };
}

macro_rules! get_xh {
    ($op:expr, $yh:expr, $kh:expr) => {
        $op.stride * ($yh - 1) - 2 * $op.pad + ($op.dilation * ($kh - 1) + 1)
    };
}

macro_rules! get_yw {
    ($op:expr, $xw:expr, $kw:expr) => {
        ($xw + 2 * $op.pad - ($op.dilation * ($kw - 1) + 1)) / $op.stride + 1
    };
}

macro_rules! get_yh {
    ($op:expr, $xh:expr, $kh:expr) => {
        ($xh + 2 * $op.pad - ($op.dilation * ($kh - 1) + 1)) / $op.stride + 1
    };
}

pub mod conv2d;
pub mod conv2d_transpose;
pub mod max_pool2d;

type MklInt = i64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    // CblasConjTrans = 113,
}

type CblasLayout = usize;

extern "C" {
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
}

#[inline]
fn cblas_sgemm_batch_wrapper(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: &[f32],
    a_array: Vec<&f32>,
    b_array: Vec<&f32>,
    beta: &[f32],
    c_array: Vec<&f32>,
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

#[test]
fn test_sgemm_batch_trans_a() {
    let batch = 2;
    let w = vec![0., 1., 2., 3., 4., 5.]; // (2, 3)
    let x = vec![0., 1., 2., 3., 4., 5., 6., 7.]; // (2, 2, 2)
    let z = alloc_uninitialized_buf(12); // (2, 2, 2)
    let m = 3; // row of op(a)
    let n = 2; // col of op(b)
    let k = 2; // col of op(a)
    cblas_sgemm_batch_wrapper(
        true,
        false,
        m,
        n,
        k,
        &[1.],              // alpha
        vec![&w[0], &w[0]], // a
        get_region_heads(batch, &x),
        &[0.], // beta
        get_region_heads(batch, &z),
        1,
        batch,
    );
    assert_eq!(
        z,
        vec![6., 9., 8., 13., 10., 17., 18., 21., 28., 33., 38., 45.]
    );
}

#[test]
fn test_sgemm_batch() {
    let batch = 2;
    let x = vec![0., 1., 2., 3.]; // (2, 2)
    let y = vec![0., 1., 2., 3., 4., 5., 6., 7.]; // (2, 2, 2)
    let z = alloc_uninitialized_buf(8); // (2, 2, 2)

    cblas_sgemm_batch_wrapper(
        false,
        false,
        2,                  // m
        2,                  // n
        2,                  // k
        &[1.],              // alpha
        vec![&x[0], &x[0]], // a
        get_region_heads(batch, &y),
        &[0.], // beta
        get_region_heads(batch, &z),
        1,
        batch,
    );
    assert_eq!(z, vec![2., 3., 6., 11., 6., 7., 26., 31.]);
}

#[inline]
fn get_region_heads<'a>(size: usize, slice: &'a [f32]) -> Vec<&'a f32> {
    let size_per_batch = slice.len() / size;
    let mut ret = Vec::with_capacity(size_per_batch);
    for i in 0..size {
        ret.push(&slice[i * size_per_batch]);
    }
    ret
}

#[link(name = "conv")]
#[no_mangle]
extern "C" {
    fn im2col_cpu(
        data_im: *const c_float,
        channels: c_int,
        height: c_int,
        width: c_int,
        kernel_h: c_int,
        kernel_w: c_int,
        pad_h: c_int,
        pad_w: c_int,
        stride_h: c_int,
        stride_w: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        data_col: *const c_float,
    );

    fn col2im_cpu(
        data_col: *const c_float,
        channels: c_int,
        height: c_int,
        width: c_int,
        kernel_h: c_int,
        kernel_w: c_int,
        pad_h: c_int,
        pad_w: c_int,
        stride_h: c_int,
        stride_w: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        data_im: *const c_float,
    );

    fn max_pool_cpu_unbatched(
        input: *const c_float,
        pad: c_int,
        h: c_int,
        w: c_int,
        out_h: c_int,
        out_w: c_int,
        c: c_int,
        b: c_int,
        size: c_int,
        stride: c_int,
        output: *const c_float,
        argmax: *const c_float,
        float_min: c_float,
    );

    #[allow(dead_code)]
    fn max_pool_cpu(
        input: *const c_float,
        pad: c_int,
        h: c_int,
        w: c_int,
        out_h: c_int,
        out_w: c_int,
        c: c_int,
        batch: c_int,
        size: c_int,
        stride: c_int,
        output: *const c_float,
        argmax: *const c_float,
        float_min: c_float,
    );

    fn max_pool_grad_cpu(
        input: *const c_float,
        h: c_int,
        w: c_int,
        c: c_int,
        batch: c_int,
        gx: *const c_float,
        argmax: *const c_float,
    );

    fn max_pool_grad_grad_cpu(
        ggx: *const c_float,
        h: c_int,
        w: c_int,
        c: c_int,
        batch: c_int,
        ggy: *const c_float,
        argmax: *const c_float,
    );
}

#[inline(always)]
fn max_pool_unbatched(
    input: &c_float,
    pad: usize,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    c: usize,
    b: usize,
    size: usize,
    stride: usize,
    output: &c_float,
    argmax: &c_float,
) {
    unsafe {
        max_pool_cpu_unbatched(
            input as *const _,
            pad as c_int,
            h as c_int,
            w as c_int,
            out_h as c_int,
            out_w as c_int,
            c as c_int,
            b as c_int,
            size as c_int,
            stride as c_int,
            output as *const _,
            argmax as *const _,
            f32::MIN,
        )
    }
}

#[inline]
#[allow(dead_code)]
fn max_pool(
    input: &c_float,
    pad: usize,
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    c: usize,
    batch: usize,
    size: usize,
    stride: usize,
    output: &c_float,
    argmax: &c_float,
) {
    unsafe {
        max_pool_cpu(
            input as *const _,
            pad as c_int,
            h as c_int,
            w as c_int,
            out_h as c_int,
            out_w as c_int,
            c as c_int,
            batch as c_int,
            size as c_int,
            stride as c_int,
            output as *const _,
            argmax as *const _,
            f32::MIN,
        )
    }
}

#[inline]
fn max_pool_grad(
    gy: &c_float,
    h: usize,
    w: usize,
    c: usize,
    batch: usize,
    gx: &c_float,
    argmax: &c_float,
) {
    unsafe {
        max_pool_grad_cpu(
            gy as *const _,
            h as c_int,
            w as c_int,
            c as c_int,
            batch as c_int,
            gx as *const c_float,
            argmax as *const c_float,
        )
    }
}

#[inline]
fn max_pool_grad_grad(
    ggx: &c_float,
    h: usize,
    w: usize,
    c: usize,
    batch: usize,
    ggy: &c_float,
    argmax: &c_float,
) {
    unsafe {
        max_pool_grad_grad_cpu(
            ggx as *const _,
            h as c_int,
            w as c_int,
            c as c_int,
            batch as c_int,
            ggy as *const c_float,
            argmax as *const c_float,
        )
    }
}

#[inline]
fn im2col(
    data_im: &c_float,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    data_col: &c_float,
) {
    unsafe {
        im2col_cpu(
            data_im as *const c_float,
            channels as i32,
            height as i32,
            width as i32,
            kernel_h as i32,
            kernel_w as i32,
            pad_h as i32,
            pad_w as i32,
            stride_h as i32,
            stride_w as i32,
            dilation_h as i32,
            dilation_w as i32,
            data_col,
        )
    }
}

#[inline]
fn col2im(
    data_col: &f32,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    data_im: &f32,
) {
    unsafe {
        col2im_cpu(
            data_col as *const c_float,
            channels as c_int,
            height as c_int,
            width as c_int,
            kernel_h as c_int,
            kernel_w as c_int,
            pad_h as c_int,
            pad_w as c_int,
            stride_h as c_int,
            stride_w as c_int,
            dilation_h as c_int,
            dilation_w as c_int,
            data_im as *const c_float,
        )
    }
}

#[inline]
fn alloc_uninitialized_buf(size: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(size);
    unsafe {
        buf.set_len(size);
    }
    buf
}

#[inline]
fn sgemm(
    trans_a: bool,
    trans_b: bool,
    a: &f32,
    b: &f32,
    c: &f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let rsa = if trans_a { 1 } else { k };
    let csa = if trans_a { m } else { 1 };
    let rsb = if trans_b { 1 } else { n };
    let csb = if trans_b { k } else { 1 };
    let rsc = n;
    let csc = 1;
    unsafe {
        let c: *mut f32 = mem::transmute(c);
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a as *const f32,
            rsa as isize,
            csa as isize,
            b as *const f32,
            rsb as isize,
            csb as isize,
            beta,
            c as *mut f32,
            rsc as isize,
            csc as isize,
        )
    }
}

#[test]
fn test_gemm_trans_a() {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [1., 2., 3., 4.];
    let c = [0.; 6];
    let m = 3; // row of op(a)
    let n = 2; // col of op(b)
    let k = 2; // col of op(a)
    sgemm(true, false, &a[0], &b[0], &c[0], m, n, k, 1., 0.);
    assert_eq!(&c, &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
}

#[test]
fn test_gemm_trans_b() {
    let a = [1., 2., 3., 4.];
    let b = [1., 2., 3., 4., 5., 6.];
    let c = [0.; 6];
    let m = 2; // row of op(a)
    let n = 3; // col of op(b)
    let k = 2; // col of op(a)
    sgemm(false, true, &a[0], &b[0], &c[0], m, n, k, 1., 0.);
    assert_eq!(&c, &[5., 11., 17., 11., 25., 39.]);
}

#[test]
fn test_conv_filter_grad() {
    use op::Op;
    let op = conv2d::Conv2DFilterGrad {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;

    let x = ::ndarray_ext::ones(&[batch_size, yh, yw, kh, kw, xch]);
    let g = ::ndarray_ext::ones(&[batch_size, ych, yh, yw]);
    let w = ::ndarray_ext::ones(&[ych, xch, kh, kw]);

    let ret = op.compute(::runtime::OpComputeContext::new(
        &::ops::zeros(&[0]), // dummy (not used)
        vec![&x, &g, &w],
    ));

    assert_eq!(w.shape(), ret[0].as_ref().unwrap().shape()); // (2, 3, 2, 2)
    assert_eq!(ret[0].clone().unwrap().into_raw_vec(), vec![8.; 24]);
}

#[test]
fn test_sequential_sgemm() {
    let x = [0., 1., 2., 3.];
    let y = [0., 1., 2., 3.];
    let z = [0.; 8];

    for i in 0..2 {
        sgemm(false, false, &x[0], &y[0], &z[i * 4], 2, 2, 2, 1., 0.)
    }
    assert_eq!([2.0, 3.0, 6.0, 11.0, 2.0, 3.0, 6.0, 11.0], z);
}

#[test]
fn test_sgemm_acc() {
    let x = [0., 1., 2., 3.];
    let y = [0., 1., 2., 3.];
    let z = [0.; 4];

    let num_iter = 3.;

    for _ in 0..num_iter as usize {
        sgemm(false, false, &x[0], &y[0], &z[0], 2, 2, 2, 1., 1.)
    }
    assert_eq!(
        [2. * num_iter, 3. * num_iter, 6. * num_iter, 11. * num_iter],
        z
    );
}

#[test]
fn test_max_pool_cpu() {
    let x = vec![0., 1., 2., 5., 4., 3., 6., 7., 8.];
    let output = alloc_uninitialized_buf(4);
    let argmax = alloc_uninitialized_buf(4);
    max_pool(
        &x[0], 0, // pad
        3, 3, // h, w
        2, 2, // out_h, out_w
        1, // c
        1, // batch
        2, // size
        1, // stride
        &output[0], &argmax[0],
    );
    assert_eq!(output, vec![5., 4., 7., 8.]);
    assert_eq!(argmax, vec![3., 4., 7., 8.]);
}
