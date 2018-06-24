extern crate ndarray;
extern crate libc;
extern crate rayon;
#[cfg(feature="blas")]
extern crate cblas_sys;
#[cfg(feature="blas")]
extern crate openblas_src;
#[cfg(not(feature="blas"))]
extern crate matrixmultiply;
#[allow(unused_imports)]
use self::rayon::iter::*;
use self::libc::{c_float, c_int};
use ndarray_ext::NdArray;
use std::mem;
use std::slice;
use std::f32;
use tensor::Tensor;

macro_rules! get_xw {
    ($op:expr, $yw:expr, $kw:expr) => {
        $op.stride_w * ($yw - 1) - $op.pad_w + ($op.dilation_w * ($kw - 1) + 1)
    };
}

macro_rules! get_xh {
    ($op:expr, $yh:expr, $kh:expr) => {
        $op.stride_h * ($yh - 1) - $op.pad_h + ($op.dilation_h * ($kh - 1) + 1)
    };
}

macro_rules! get_yw {
    ($op:expr, $xw:expr, $kw:expr) => {
        ($xw + 2 * $op.pad_w - ($op.dilation_w * ($kw - 1) + 1)) / $op.stride_w + 1
    };
}

macro_rules! get_yh {
    ($op:expr, $xh:expr, $kh:expr) => {
        ($xh + 2 * $op.pad_h - ($op.dilation_h * ($kh - 1) + 1)) / $op.stride_h + 1
    };
}

// Returns: &Vec<f32>
macro_rules! get_or_insert_cols {
    ($me:expr, $batch_size:expr, $num_elements_in_batch_c:expr) => {
        unsafe {
            let slf: &mut Self = mem::transmute($me);
            let cols: &Vec<f32> = mem::transmute(
                slf.cols.get_or_insert_with(||
                    alloc_uninitialized_buf($batch_size * $num_elements_in_batch_c)
                )
            );
            cols
        }
    };
}

pub mod conv2d;
pub mod conv2d_transpose;
pub mod max_pool;

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
        float_min: c_float
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

#[inline]
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
            input as *const _, pad as c_int,
            h as c_int, w as c_int, out_h as c_int, out_w as c_int, c as c_int, batch as c_int,
            size as c_int, stride as c_int,
            output as *const _, argmax as *const _, f32::MIN
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
fn alloc_uninitialized_buf(size: usize) -> Vec<f32>
{
    let mut buf = Vec::with_capacity(size);
    unsafe {
        buf.set_len(size);
    }
    buf
}


#[inline]
fn sgemm(trans_a: bool, trans_b: bool,
         a: &f32, b: &f32, c: &f32,
         m: usize, n: usize, k: usize, alpha: f32, beta: f32)
{
    #[cfg(feature="blas")] {
        let m = m as i32;
        let n = n as i32;
        let k = k as i32;
        unsafe {
            cblas_sys::cblas_sgemm(
                cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
                if trans_a {
                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans
                } else {
                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans
                },
                if trans_b {
                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans
                } else {
                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans
                },
                m, n, k,
                alpha,
                a as *const f32, if trans_a { m } else { k }, // lda
                b as *const f32, if trans_b { k } else { n }, // ldb
                beta,
                mem::transmute::<&f32, *mut f32>(c), n, // ldc
            );
        }
    }
    #[cfg(not(feature="blas"))] {
        let rsa = if trans_a { 1 } else { k };
        let csa = if trans_a { m } else { 1 };
        let rsb = if trans_b { 1 } else { n };
        let csb = if trans_b { k } else { 1 };
        let rsc = n;
        let csc = 1;
        unsafe {
            let c: *mut f32 = mem::transmute(c);
            matrixmultiply::sgemm(m, k, n,
                                  alpha,
                                  a as *const f32, rsa as isize, csa as isize,
                                  b as *const f32, rsb as isize, csb as isize,
                                  beta,
                                  c as *mut f32, rsc as isize, csc as isize
            )
        }
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
fn test_conv_filter_grad()
{
    use ::op::Op;
    let op = conv2d::Conv2DFilterGrad {
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;

    let x = ::ndarray_ext::ones(&[batch_size, yh, yw, kh, kw, xch]);
    let g = ::ndarray_ext::ones(&[batch_size, ych, yh, yw]);
    let w = ::ndarray_ext::ones(&[ych, xch, kh, kw]);

    let ret = op.compute(::runtime::OpComputeContext {
        xs: vec![&x, &g, &w],
        node: &::ops::zeros(&[0]) // dummy (not used)
    });

    assert_eq!(w.shape(), ret[0].as_ref().unwrap().shape());  // (2, 3, 2, 2)
    assert_eq!(ret[0].clone().unwrap().into_raw_vec(), vec![8.; 24]);
}

#[test]
fn test_sequential_sgemm()
{
    let x = [0., 1., 2., 3.];
    let y = [0., 1., 2., 3.];
    let z = [0.; 8];

    for i in 0..2 {
        sgemm(false, false, &x[0], &y[0], &z[i * 4], 2, 2, 2, 1., 0.)
    }
    assert_eq!([2.0, 3.0, 6.0, 11.0, 2.0, 3.0, 6.0, 11.0], z);
}

#[test]
fn test_sgemm_acc()
{
    let x = [0., 1., 2., 3.];
    let y = [0., 1., 2., 3.];
    let z = [0.; 4];

    let num_iter = 3.;

    for _ in 0..num_iter as usize {
        sgemm(false, false, &x[0], &y[0], &z[0], 2, 2, 2, 1., 1.)
    }
    assert_eq!([2.*num_iter, 3.*num_iter, 6.*num_iter, 11.*num_iter], z);
}

#[test]
fn test_max_pool_cpu()
{
    let x = vec![
        0., 1., 2.,
        5., 4., 3.,
        6., 7., 8.
    ];
    let output = alloc_uninitialized_buf(4);
    let argmax = alloc_uninitialized_buf(4);
    max_pool(&x[0], 0, // pad
                     3, 3, // h, w
                     2, 2, // out_h, out_w
                     1, // c
                     1, // batch
                     2, // size
                     1, // stride
                     &output[0], &argmax[0]);
    assert_eq!(output, vec![5., 4., 7., 8.]);
    assert_eq!(argmax, vec![3., 4., 7., 8.]);
}
