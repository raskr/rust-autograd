extern crate ndarray;
extern crate libc;
extern crate rayon;
extern crate cblas;
extern crate cblas_sys;
extern crate openblas_src;

//#[cfg(feature="blas")]
//#[cfg(feature="blas")]

use self::cblas::Layout;
use self::rayon::iter::*;
use self::libc::{c_float, c_int};
use ndarray_ext::NdArray;
use ::op::Op;
use std::mem;
use tensor::Tensor;
//use self::blas as mm;

#[link(name = "conv")]
#[no_mangle]
extern "C" {

    fn im2col_kernel(
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

    fn col2im_kernel(
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
}

#[inline]
fn exec_im2col(
    data_im: &c_float,
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
) {
    unsafe {
        im2col_kernel(
            data_im as *const c_float,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            data_col,
        )
    }
}

#[inline]
fn exec_col2im(
    data_col: &c_float,
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
    data_im: &c_float,
) {
    unsafe {
        col2im_kernel(
            data_col,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            data_im as *const c_float,
        )
    }
}


#[inline]
fn _gemm(trans_a: bool,
         trans_b: bool,
         m: c_int,
         n: c_int,
         k: c_int,
         alpha: c_float,
         a: *const c_float,
         lda: c_int,
         b: *const c_float,
         ldb: c_int,
         beta: c_float,
         c: *const c_float,
         ldc: c_int,
)
{
    use std::slice;
    unsafe {
        let c = ::std::mem::transmute::<*const c_float, *mut c_float>(c);
        let a = slice::from_raw_parts(a, (m * k) as usize);
        let b = slice::from_raw_parts(b, (k * n) as usize);
        let c = slice::from_raw_parts_mut(c, (m * n) as usize);
        cblas::sgemm(Layout::RowMajor,
                     if trans_a { cblas::Transpose::Ordinary } else { cblas::Transpose::None },
                     if trans_b { cblas::Transpose::Ordinary } else { cblas::Transpose::None },
                     m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        );
//        println!("{:?}", Vec::from_raw_parts(c_, 6, 6));
    };
}

//enum GemmTrans {
//    N = 78,
//    T = 84
//}

//#[inline]
//fn gemm(
//    transa: u8,
//    transb: u8,
//    m: i32,
//    n: i32,
//    k: i32,
//    alpha: f32,
//    a: &[f32],  // lhs
//    lda: i32,   // 1
//    b: &[f32],  // rhs
//    ldb: i32,   //
//    beta: f32,  // used for init?
//    c: &mut [f32],  // dst
//    ldc: i32,
//)
//{
//    unsafe {
//        mm::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
//    }
//}

type Mat = ndarray::Array2<f32>;
type Array4 = ndarray::Array4<f32>;

pub struct Conv2DGradRHS {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

pub struct Conv2D {
    pub pad_h: i32,
    pub pad_w: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    pub dilation_h: i32,
    pub dilation_w: i32,
}

pub struct Conv2DTransposed {
    pub pad_h: i32,
    pub pad_w: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    pub dilation_h: i32,
    pub dilation_w: i32,
}

#[inline]
fn alloc_uninitialized_cols(xch: i32, kw: i32, kh: i32, yh: i32, yw: i32) -> Vec<f32>
{
    let len = (xch * kw * kh * yh * yw) as usize;
    let mut cols = Vec::with_capacity(len);
    unsafe {
        cols.set_len(len);
    }
    cols
}

#[test]
fn test_tensor_size_after_convolution()
{
    let op = Conv2D {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);

    let yh = get_yh(&op, xh, kh);
    let yw = get_yw(&op, xw, kw);
    assert_eq!(yh, 2);
    assert_eq!(yw, 2);
}

#[test]
fn test_tensor_size_after_convolution_t()
{
    let op = Conv2DTransposed {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh(&op, yh, kh);
    let xw = get_xw(&op, yw, kw);
    assert_eq!(xh, 3);
    assert_eq!(xw, 3);
}

#[test]
fn test_col2im()
{
    let op = Conv2DTransposed {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };
    let xch = 2;
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh(&op, yh, kh);
    let xw = get_xw(&op, yw, kw);

    // 32 elem
    let col = (0..(xch * kh * kw * yh * yw) as usize).map(|a| a as f32).collect::<Vec<_>>();
    let im = vec![0.; 19]; // +1

    unsafe {
        col2im_kernel(col.as_ptr(),
                      xch,
                      xh,
                      xw,
                      kh,
                      kw,
                      op.pad_h,
                      op.pad_w,
                      op.stride_h,
                      op.stride_w,
                      op.dilation_h,
                      op.dilation_w,
                      im.as_ptr());
    }

    assert_eq!(im[18], 0.);
    assert_ne!(im[17], 0.);
}

#[inline]
fn sgemm(trans_a: bool, trans_b: bool,
         a: *const f32, b: *const f32, c: *const f32,
         m: i32, n: i32, k: i32)
{
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
            1.,
            a, if trans_a { m } else { k },
            b, if trans_a { k } else { n },
            0.,
            mem::transmute::<*const f32, *mut f32>(c), n,
        );
    }
}

#[test]
fn test_gemm_trans_a() {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [1., 2., 3., 4.];
    let c = [0.; 6];
    let m = 3i32; // row of op(a)
    let n = 2i32; // col of op(b)
    let k = 2i32; // col of op(a)
    sgemm(true, false, a.as_ptr(), b.as_ptr(), c.as_ptr(), m, n, k);
    assert_eq!(&c, &[13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
}

#[test]
fn test_gemm_trans_b() {
    let a = [1., 2., 3., 4.];
    let b = [1., 2., 3., 4., 5., 6.];
    let c = [0.; 6];
    let m = 2i32; // row of op(a)
    let n = 3i32; // col of op(b)
    let k = 2i32; // col of op(a)
    sgemm(false, true, a.as_ptr(), b.as_ptr(), c.as_ptr(), m, n, k);
    assert_eq!(&c, &[5., 11., 17., 11., 25., 39.]);
}


#[test]
fn test_deconv2d_unbatched()
{
    let op = Conv2DTransposed {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let kh = 2;
    let kw = 2;
    let ych = 1;
    let xch = 2;
    let yh = 2;
    let yw = 2;
    let xh = get_xh(&op, yh, kh);
    let xw = get_xw(&op, yw, kw);
    let k = ych;
    let n = yh * yw;
    let m = kh * kw * xch;

    // 1. gemm
    let w = vec![1.; (kh * kw * xch * ych) as usize];
    let gy = vec![1.; (ych * xw * xh) as usize];
    let col = vec![0.; (kh * kw * xch * yh * yw) as usize];
    sgemm(true, false, w.as_ptr(), gy.as_ptr(), col.as_ptr(), m, n, k);

    // 2. col2im
    let im = vec![0.; (xch * xh * xw) as usize + 1];
    unsafe {
        col2im_kernel(col.as_ptr(), xch, xh, xw, kh, kw,
                      op.pad_h, op.pad_w, op.stride_h, op.stride_w,
                      op.dilation_h, op.dilation_w, im.as_ptr());
    }
    assert_eq!(im[(xch * xh * xw) as usize], 0.);
    assert_eq!(im[(xch * xh * xw) as usize - 1], 1.);
}

#[test]
fn test_deconv()
{
    let op = Conv2DTransposed {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let (xh, xw) = (get_xh(&op, yh, kh) as usize, get_xw(&op, yw, kw) as usize);
    let batch_size = 2;

    let x = ::ndarray_ext::zeros(&[batch_size, xch, xh, xw]);
    let w = ::ndarray_ext::zeros(&[ych, xch, kh as usize, kw as usize]);
    let g = ::ndarray_ext::zeros(&[batch_size, ych, yh as usize, yw as usize]);

    let ret = op.compute(::runtime::OpComputeContext {
        xs: vec![&g, &w],
        node: &::ops::zeros(&[0]) // dummy (not used)
    });

    assert_eq!(x.shape(), ret[0].as_ref().unwrap().shape());
}

impl ::op::Op for Conv2DTransposed {
    fn name(&self) -> &str
    {
        "Conv2DTransposed"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();

        let gy: &NdArray = xs[0];  // (batch, ych, yh, yw)
        let gy_shape = gy.shape();
        let w: &NdArray = xs[1];      // (ych, xch, kh, kw)
        let w_shape = w.shape();

        // slice for random access
        let gy = gy.as_slice().expect("Not standard layout");

        let batch_size = gy_shape[0];
        let ych = gy_shape[1] as i32;
        let yh = gy_shape[2] as i32;
        let yw = gy_shape[3] as i32;

        let xch = w_shape[1] as i32;
        let kh = w_shape[2] as i32;
        let kw = w_shape[3] as i32;
        let xh = get_xh(self, yh, kh);
        let xw = get_xw(self, yw, kw);

        let k = ych;
        let n = yh * yw;
        let m = kh * kw * xch;

        let num_elements_in_batch = (ych * yh * yw) as usize;
        let im_shape = (1, xch as usize, xh as usize, xw as usize);

        let w: &f32 = unsafe { &*w.as_ptr() };
        // w: (ych, xch, kh, kw)

        let dots: Vec<_> = (0..batch_size).into_par_iter().map(move |i| { // for each mini-batch
            // (ych, yh, yw)
            let g = &gy[i * num_elements_in_batch];

            // alloc buffers
            let mut col = Vec::with_capacity((kh * kw * xch * yh * yw) as usize);
            let mut im = Vec::with_capacity((xch * xh * xw) as usize);
            unsafe {
                col.set_len((kh * kw * xch * yh * yw) as usize);
                im.set_len((xch * xh * xw) as usize);
            }

            // gemm (xch * kh * kw, ych) x (ych, yh * yw)
            sgemm(true, false, w as *const f32,
                  g as *const f32,
                  col.as_ptr(), m, n, k);

            // col2im
            exec_col2im(&col[0], ych, yh, yw, kh, kw,
                          self.pad_h, self.pad_w,
                          self.stride_h, self.stride_w,
                          self.dilation_h, self.dilation_w, &im[0]);
            Array4::from_shape_vec(im_shape, im).unwrap()
        }).collect::<Vec<_>>();

        // Stack results (batch, num_filters, out_h * out_w)
        let stack = ndarray::stack(ndarray::Axis(0),
                                   &dots.iter().map(|d| d.view()).collect::<Vec<_>>()).unwrap();

        // Reshape it into (batch, yc, yh, yw)
        let ret = stack.into_shape(ndarray::IxDyn(
            &[batch_size, xch as usize, xh as usize, xw as usize])).unwrap();

        vec![Ok(ret)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}


impl ::op::Op for Conv2DGradRHS {
    fn name(&self) -> &str
    {
        "Conv2DGradRHS"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let col = xs[0];  // must be columns
        let gy = xs[1];
        let w = xs[2];

        let w_shape = w.shape();
        let gy_shape = gy.shape();
        let (xch, kh, kw) = (x_shape[1], x_shape[2], x_shape[3]);
        let (batch_size, ych) = (gy_shape[0], gy_shape[1]);

        let m = ych as isize;
        let n = (kh * kw * xch) as isize;
        let k = batch_size * xh * xw;

        let xh = get_xh_(&op, yh, kh) as usize;
        let xw = get_xw_(&op, yw, kw) as usize;

        // gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
        let im = {
            let len = ych * kh * kw * xch;
            let mut buf = Vec::with_capacity(len);
            unsafe {
                buf.set_len(len);
            }
            buf
        };
        sgemm(false, true, gy, col, im, m, n, k);
        vec![Ok(NdArray::from_shape_vec(
            ndarray::IxDyn(&[ych, xch, kh, kw]), im).unwrap())]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}

impl ::op::Op for Conv2D {
    fn name(&self) -> &str
    {
        "Conv2D"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let x: &NdArray = xs[0];
        let w = xs[1].view();

        let (batch_size, xch, xh, xw) = {
            let x_shape = x.shape();
            (x_shape[0], x_shape[1] as i32, x_shape[2] as i32, x_shape[3] as i32)
        };

        let (ych, kh, kw) = {
            let kernel_shape = w.shape();
            (kernel_shape[0] as i32, kernel_shape[2] as i32, kernel_shape[3] as i32)
        };

        let yh = get_yh(self, xh, kh);
        let yw = get_yw(self, xw, kw);

        let num_elements_in_batch = (xch * xh * xw) as usize;
        let w = w.into_shape((ych as usize, (xch * kh * kw) as usize)).unwrap();
        let final_shape = ((xch * kh * kw) as usize, (yh * yw) as usize);

        // Stream of "im2col + matrix mul" is executed in parallel (with rayon thread pool).
        let dots: Vec<_> = (0..batch_size).into_par_iter().map(move |i| { // for each mini-batch

            let cols = alloc_uninitialized_cols(xch, kw, kh, yh, yw);

            exec_im2col(
                &x.as_slice().expect("Not standard layout")[i * num_elements_in_batch],
                xch, xh, xw, kh, kw,
                self.pad_h, self.pad_w,
                self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w,
                cols.as_ptr()
            );

            // (kkc, wh)
            let col: Mat = Mat::from_shape_vec(final_shape, cols).unwrap();

            // Dot product and then expand the batch dim.
            let dot = {
                let dot = w.dot(&col);
                let (a, b) = {
                    let shape = dot.shape();
                    (shape[0], shape[1])
                };
                dot.into_shape((1, a, b)).unwrap() // safe unwrap
            };
            dot
        }).collect();

        // Stack results (batch, num_filters, out_h * out_w)
        let stack = ndarray::stack(ndarray::Axis(0),
                                   &dots.iter().map(|d| d.view()).collect::<Vec<_>>()).unwrap();

        // Reshape it into (batch, yc, yh, yw)
        let ret = stack.into_shape(ndarray::IxDyn(
            &[batch_size, ych as usize, yh as usize, yw as usize])).unwrap();

        vec![Ok(ret)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}

#[test]
fn test_parallel_im2col()
{
    let op = Conv2D {
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let batch_size = 2usize;
    let xch = 2i32;
    let (xh, xw) = (3i32, 3i32);
    let (kh, kw) = (2i32, 2i32);
    let num_elements_in_batch = (xch * xh * xw) as usize;
    let yh = get_yh(&op, xh, kh);
    let yw = get_yw(&op, xw, kw);
    let x = (0..(batch_size as i32 * xch * xw * xh)).map(|a| a as f32).collect::<Vec<_>>();

    // Call im2col on 2 chunks in parallel.
    let cols_seq = (0..batch_size).into_par_iter().map(move |i| { // for each mini-batch
        let cols = alloc_uninitialized_cols(xch, kw, kh, yh, yw);
        exec_im2col(
            &x[i * num_elements_in_batch],
            xch, xh, xw, kh, kw,
            op.pad_h, op.pad_w,
            op.stride_h, op.stride_w,
            op.dilation_h, op.dilation_w,
            cols.as_ptr()
        );
        cols
    }).collect::<Vec<Vec<f32>>>();

    assert_eq!(
        cols_seq[0],
        vec![
            0.0, 1.0, 3.0, 4.0,
            1.0, 2.0, 4.0, 5.0,
            3.0, 4.0, 6.0, 7.0,
            4.0, 5.0, 7.0, 8.0,

            9.0, 10.0, 12.0, 13.0,
            10.0, 11.0, 13.0, 14.0,
            12.0, 13.0, 15.0, 16.0,
            13.0, 14.0, 16.0, 17.0
        ]
    );

    assert_eq!(
        cols_seq[1],
        vec![
            18.0, 19.0, 21.0, 22.0,
            19.0, 20.0, 22.0, 23.0,
            21.0, 22.0, 24.0, 25.0,
            22.0, 23.0, 25.0, 26.0,
            27.0, 28.0, 30.0, 31.0,
            28.0, 29.0, 31.0, 32.0,
            30.0, 31.0, 33.0, 34.0,
            31.0, 32.0, 34.0, 35.0
        ]
    );

}

#[test]
fn test_im2col()
{
    let op = Conv2D {
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let xch = 2;
    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);
    let yh = get_yh(&op, xh, kh);
    let yw = get_yw(&op, xw, kw);

    let x = ndarray::Array1::range(0., (xch * xw * xh) as f32, 1.)
        .into_shape((1, xch as usize, xw as usize, xh as usize))
        .unwrap();

    let cols = alloc_uninitialized_cols(xch, kw, kh, yh, yw);

    unsafe {
        im2col_kernel(x.as_ptr(),
                      xch as i32,
                      xh as i32, xw as i32,
                      kh as i32, kw as i32,
                      op.pad_h as i32, op.pad_w as i32,
                      op.stride_h as i32, op.stride_w as i32,
                      op.dilation_h as i32, op.dilation_w as i32,
                      cols.as_ptr())
    };

    assert_eq!(
        cols,
        vec![
            0.0, 1.0, 3.0, 4.0,
            1.0, 2.0, 4.0, 5.0,
            3.0, 4.0, 6.0, 7.0,
            4.0, 5.0, 7.0, 8.0,
            9.0, 10.0, 12.0, 13.0,
            10.0, 11.0, 13.0, 14.0,
            12.0, 13.0, 15.0, 16.0,
            13.0, 14.0, 16.0, 17.0
        ]
    )
}

#[test]
fn test_conv2d() {

    let op = Conv2D {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
    };

    let x = ndarray::Array1::range(0., 1.*2.*3.*3., 1.)
        .into_shape((1, 2, 3, 3)).unwrap().into_dyn();
    let w = ::ndarray_ext::zeros(&[/*out_ch=*/1, /*in_ch=*/2, /*row=*/2, /*col=*/2]);

    let ret = op.compute(::runtime::OpComputeContext {
        xs: vec![&x, &w],
        node: &::ops::zeros(&[0]) // dummy (not used)
    });
    let x_shape = x.shape();
    assert_eq!(
        ret[0].as_ref().unwrap().shape(),
        &[x_shape[0], 1, 2, 2]
    );
}

#[inline]
fn get_xh_(op: &Conv2DGradRHS, yh: usize, kh: usize) -> usize
{
    op.stride_h * (yh - 1) - op.pad_h + (op.dilation_h * (kh - 1) + 1)
}

#[inline]
fn get_xw_(op: &Conv2DGradRHS, yw: i32, kw: i32) -> usize
{
    op.stride_w * (yw - 1) - op.pad_w + (op.dilation_w * (kw - 1) + 1)
}

#[inline]
fn get_xh(op: &Conv2DTransposed, yh: i32, kh: i32) -> i32
{
    op.stride_h * (yh - 1) - op.pad_h + (op.dilation_h * (kh - 1) + 1)
}

#[inline]
fn get_xw(op: &Conv2DTransposed, yw: i32, kw: i32) -> i32
{
    op.stride_w * (yw - 1) - op.pad_w + (op.dilation_w * (kw - 1) + 1)
}

#[inline]
fn get_yw(op: &Conv2D, xw: i32, kw: i32) -> i32
{
    (xw + 2 * op.pad_w - (op.dilation_w * (kw - 1) + 1)) / op.stride_w + 1
}

#[inline]
fn get_yh(op: &Conv2D, xh: i32, kh: i32) -> i32
{
    (xh + 2 * op.pad_h - (op.dilation_h * (kh - 1) + 1)) / op.stride_h + 1
}
