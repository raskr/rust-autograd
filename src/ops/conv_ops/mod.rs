extern crate ndarray;
extern crate libc;
extern crate rayon;
extern crate cblas_sys;
extern crate openblas_src;

use self::rayon::iter::*;
use self::libc::{c_float, c_int};
use ndarray_ext::NdArray;
use std::mem;
use std::slice;
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
        im2col_kernel(
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
fn exec_col2im(
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
        col2im_kernel(
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

pub struct Conv2DFilterGrad {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

pub struct Conv2DTransposeFilterGrad {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
}

pub struct Conv2D {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
}

pub struct Conv2DWithCols {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
}

pub struct Conv2DTranspose {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
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
        cols: None,
    };

    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);

    let yh = get_yh!(&op, xh, kh);
    let yw = get_yw!(&op, xw, kw);
    assert_eq!(yh, 2);
    assert_eq!(yw, 2);
}

#[test]
fn test_tensor_size_after_convolution_t()
{
    let op = Conv2DTranspose {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
        cols: None,
    };
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh!(&op, yh, kh);
    let xw = get_xw!(&op, yw, kw);
    assert_eq!(xh, 3);
    assert_eq!(xw, 3);
}

#[test]
fn test_parallel_col2im()
{
    let batch_size = 2;
    let op = Conv2DTranspose {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
        cols: None,
    };
    let xch = 3;
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh!(&op, yh, kh);
    let xw = get_xw!(&op, yw, kw);

    let num_elements_in_batch_col = xch * kh * kw * yh * yw;
    let num_elements_in_batch_im = xch * xh * xw;
    let cols = vec![2f32; 108*batch_size];
    let im = vec![0f32; batch_size * xch * xh * xw];

    (0..batch_size).into_par_iter().for_each(|i| {
        unsafe {
            let cols_head = (&cols[i * num_elements_in_batch_col]) as *const f32;
            let im_head = (&im[i * num_elements_in_batch_im]) as *const f32;
            col2im_kernel(cols_head,
                          xch as i32,
                          xh as i32,
                          xw as i32,
                          kh as i32,
                          kw as i32,
                          op.pad_h as i32,
                          op.pad_w as i32,
                          op.stride_h as i32,
                          op.stride_w as i32,
                          op.dilation_h as i32,
                          op.dilation_w as i32,
                          im_head);
        }
    });

    assert_eq!(
        im,
        vec![
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,

            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
        ]
    );
}

#[inline]
fn sgemm(trans_a: bool, trans_b: bool,
         a: &f32, b: &f32, c: &f32,
         m: usize, n: usize, k: usize, alpha: f32, beta: f32)
{
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
            a as *const f32, if trans_a { m } else { k },
            b as *const f32, if trans_b { k } else { n },
            beta,
            mem::transmute::<&f32, *mut f32>(c), n,
        );
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
    let op = Conv2DFilterGrad {
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
fn test_deconv()
{
    use ::op::Op;
    let op = Conv2DTranspose {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
        cols: None,
    };
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let (xh, xw) = (3, 3);
    let batch_size = 2;

    let w = ::ndarray_ext::ones(&[ych, xch, kh, kw]);
    let g = ::ndarray_ext::ones(&[batch_size, ych, yh, yw]);

    let ret = op.compute(::runtime::OpComputeContext {
        xs: vec![&g, &w],
        node: &::ops::zeros(&[0]) // dummy (not used)
    });

    let x = ::ndarray_ext::ones(&[batch_size, xch, xh, xw]);
    assert_eq!(x.shape(), ret[0].as_ref().unwrap().shape());


    assert_eq!(
        ret[0].clone().unwrap().into_raw_vec(),
        vec![
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,

            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0,
        ]
    )
}

impl ::op::Op for Conv2DTranspose {
    fn name(&self) -> &str
    {
        "Conv2DTranspose"
    }

    #[allow(mutable_transmutes)]
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();

        let gy: &NdArray = xs[0];  // (batch, ych, yh, yw)
        let w: &NdArray = xs[1];   // (ych, xch, kh, kw)
        let gy_shape = gy.shape();
        let f_shape = w.shape();

        let batch_size = gy_shape[0];
        let ych = gy_shape[1];
        let yh = gy_shape[2];
        let yw = gy_shape[3];

        let xch = f_shape[1];
        let kh = f_shape[2];
        let kw = f_shape[3];
        let xh = get_xh!(self, yh, kh);
        let xw = get_xw!(self, yw, kw);

        let k = ych;
        let n = yh * yw;
        let m = kh * kw * xch;

        let num_elements_in_batch_gy = ych * yh * yw;
        let num_elements_in_batch_gx = xch * xh * xw;
        let num_elements_in_batch_col = xch * kh * kw * yh * yw;

        // Targets of gemm
        let gy = unsafe {
            slice::from_raw_parts(gy.as_ptr(), gy.len())
        };
        let w: &f32 = unsafe { &*w.as_ptr() };

        // alloc buffers as necessary
        let col = get_or_insert_cols!(self, batch_size, num_elements_in_batch_col);
        let gx = vec![0.; batch_size * num_elements_in_batch_gx];

        (0..batch_size).into_par_iter().for_each(|i| { // for each mini-batch
            let gy_region_head = &gy[i * num_elements_in_batch_gy];
            let col_region_head = &col[i * num_elements_in_batch_col];
            let gx_region_head = &gx[i * num_elements_in_batch_gx];

            sgemm(true, false, w, gy_region_head, col_region_head, m, n, k, 1., 0.);

            exec_col2im(col_region_head, xch, xh, xw, kh, kw,
                          self.pad_h, self.pad_w,
                          self.stride_h, self.stride_w,
                          self.dilation_h, self.dilation_w, gx_region_head);
        });

        let gx = NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, xch, xh, xw]), gx);
        vec![Ok(gx.unwrap())]
    }

    fn grad(&self, gy: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = xs[0];
        let w = xs[1];

        let gx1 = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                 Conv2D {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        let gx2 = Tensor::builder()
            .set_inputs(vec![gy, x, w])
            .build(
                Conv2DTransposeFilterGrad {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None
                }
            );

        vec![Some(gx1), Some(gx2)]
    }
}

impl ::op::Op for Conv2DFilterGrad {
    fn name(&self) -> &str
    {
        "Conv2DFilterGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let cols = xs[0];  // must be columns
        let gy = xs[1];
        let w = xs[2];

        let cols_shape = cols.shape();
        let gy_shape = gy.shape();
        let f_shape = w.shape();

        let num_elements_in_batch_g = {
            gy_shape[1] *
            gy_shape[2] *
            gy_shape[3]
        };

        let num_elements_in_batch_c = {
            cols_shape[1] *
            cols_shape[2] *
            cols_shape[3] *
            cols_shape[4] *
            cols_shape[5]
        };

        let (xch, kh, kw) = (f_shape[1], f_shape[2], f_shape[3]);
        let (batch_size, ych, yh, yw) = (gy_shape[0], gy_shape[1], gy_shape[2], gy_shape[3]);

        let m = ych;
        let n = kh * kw * xch;
        let k = yh * yw;

        // Prepare bufs
        let cols = unsafe {
            slice::from_raw_parts(cols.as_ptr(), cols.len())
        };
        let gy = unsafe {
            slice::from_raw_parts(gy.as_ptr(), gy.len())
        };
        let gf = alloc_uninitialized_buf(ych * xch * kh * kw);
        let gf_head = unsafe { &*gf.as_ptr() };

        for i in 0..batch_size {
            sgemm(false, true,
                  &gy[i * num_elements_in_batch_g],
                  &cols[i * num_elements_in_batch_c],
                  gf_head, m, n, k, 1., (i != 0) as i32 as f32);
        }

        println!("inputs of Conv2DFilterGrad: {:?}", ctx.node.inputs);
        println!("f_shape: {:?}", f_shape);
        println!("vec shape: {:?}", &[ych, xch, kh, kw]);

        // (2, 3, 2, 2) vs (3, 3, 2, 2). ych は 3でなく2であるべき。
        let gf = NdArray::from_shape_vec(f_shape, gf).unwrap();
        vec![Ok(gf)]
    }

    fn grad(&self, ggf: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let cols = xs[0];
        let gy = xs[1];

        let g_cols = Tensor::builder()
            .set_inputs(vec![gy, ggf])
            .build(
                 Conv2DTranspose {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        let ggy = Tensor::builder()
            .set_inputs(vec![cols, ggf])
            .build(
                Conv2DWithCols {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        vec![Some(g_cols), Some(ggy), None]
    }
}

impl ::op::Op for Conv2DTransposeFilterGrad {
    fn name(&self) -> &str
    {
        "Conv2DTransposeFilterGrad"
    }

    #[allow(mutable_transmutes)]
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let gy = xs[0];
        let x = xs[1];
        let w = xs[2];

        let x_shape = x.shape();
        let gy_shape = gy.shape();
        let f_shape = w.shape();

        let batch_size = x_shape[0];
        let (kh, kw) = (f_shape[2], f_shape[3]);

        let num_elements_in_batch_g = {
            gy_shape[1] *
            gy_shape[2] *
            gy_shape[3]
        };
        let num_elements_in_batch_c = {
            get_yh!(self, gy_shape[2], kh) *
            get_yw!(self, gy_shape[3], kw) * kh * kw * gy_shape[1]
        };
        let num_elements_in_batch_x = x_shape[1] * x_shape[2] * x_shape[3];

        let m = x_shape[1];
        let n = kh * kw * gy_shape[1];
        let k = get_yh!(self, gy_shape[2], kh) * get_yw!(self, gy_shape[3], kw);

        let x = unsafe {
            slice::from_raw_parts(x.as_ptr(), x.len())
        };
        let gy = unsafe {
            slice::from_raw_parts(gy.as_ptr(), gy.len())
        };

        // Allocate buffer as necessary
        let cols = get_or_insert_cols!(self, batch_size, num_elements_in_batch_c);

        let gw = alloc_uninitialized_buf(f_shape[0] * f_shape[1] * f_shape[2] * f_shape[3]);
        let gw_head = unsafe { &*gw.as_ptr() };

        for i in 0..batch_size {
            let x_region_head = &x[i * num_elements_in_batch_x];
            let c_region_head = &cols[i * num_elements_in_batch_c];
            let g_region_head = &gy[i * num_elements_in_batch_g];

            exec_im2col(
                g_region_head,
                gy_shape[1], gy_shape[2], gy_shape[3], kh, kw,
                self.pad_h, self.pad_w,
                self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w,
                c_region_head
            );

            sgemm(false, true,
                  x_region_head,
                  c_region_head,
                  gw_head, m, n, k, 1., (i != 0) as i32 as f32);
        }

        vec![Ok(NdArray::from_shape_vec(f_shape, gw).unwrap())]
    }

    fn grad(&self, gg: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gy = xs[0];
        let x = xs[1];

        let g1 = Tensor::builder()
            .set_inputs(vec![gy, gg])
            .build(
                Conv2DTranspose {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        let g2 = Tensor::builder()
            .set_inputs(vec![x, gg])
            .build(
                Conv2D {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        vec![Some(g1), Some(g2), None]
    }
}

impl ::op::Op for Conv2D {
    fn name(&self) -> &str
    {
        "Conv2D"
    }

    #[allow(mutable_transmutes)]
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let x: &NdArray = xs[0];
        let w: &NdArray = xs[1];

        // Extract size params
        let (batch_size, xch, xh, xw) = {
            let x_shape = x.shape();
            (x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        };
        let (ych, kh, kw) = {
            let f_shape = w.shape();
            (f_shape[0], f_shape[2], f_shape[3])
        };
        let yh = get_yh!(self, xh, kh);
        let yw = get_yw!(self, xw, kw);

        // Parameters for sgemm
        let num_elements_in_batch_x = xch * xh * xw;
        let num_elements_in_batch_y = ych * yh * yw;
        let num_elements_in_batch_c = xch * kw * kh * yh * yw;
        let m = ych;
        let n = yh * yw;
        let k = xch * kh * kw;

        // Prepare pointers to buffers
        let x = unsafe {
            slice::from_raw_parts(x.as_ptr(), batch_size * xch * xh * xw)
        };

        // alloc buffers as necessary
        let c = get_or_insert_cols!(self, batch_size, num_elements_in_batch_c);
        let y = alloc_uninitialized_buf(batch_size * num_elements_in_batch_y);
        let w: &f32 = unsafe { &*w.as_ptr() };

        // Do task
        (0..batch_size).into_par_iter().for_each(|i| { // for each batch
            // 1. im2col
            let x_region_head = &x[i * num_elements_in_batch_x];
            let c_region_head = &c[i * num_elements_in_batch_c];

            exec_im2col(
                x_region_head,
                xch, xh, xw, kh, kw,
                self.pad_h, self.pad_w,
                self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w,
                c_region_head
            );

            // 2. sgemm (y <- wc)
            let y_region_head = &y[i * num_elements_in_batch_y];

            sgemm(false, false, w, c_region_head, y_region_head, m, n, k, 1., 0.);
        });

        // Move vectors into NdArrays
        let y = NdArray::from_shape_vec(
            ndarray::IxDyn(&[batch_size, ych, yh, yw]), y).unwrap();

        let cols = NdArray::from_shape_vec(
            ndarray::IxDyn(&[batch_size, xch, kw, kh, yh, yw]), c.clone()).unwrap();

        vec![Ok(y), Ok(cols)]
    }

    fn grad(&self, gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>
    {
        let w = xs[1];

        let gx = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                Conv2DTranspose {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        let cols = &::ops::nth_tensor(y, 1);
        let gw = Tensor::builder()
            .set_inputs(vec![cols, gy, w])
            .build(
                Conv2DFilterGrad {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                }
            );

        vec![Some(gx), Some(gw)]
    }
}

impl ::op::Op for Conv2DWithCols {
    fn name(&self) -> &str
    {
        "Conv2DWithCols"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ::op::ComputeResult
    {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let cols: &NdArray = xs[0];
        let w: &NdArray = xs[1];

        // Extract size params
        let cols_shape = cols.shape();
        let f_shape = w.shape();
        let (ych, xch, kh, kw) = {
            (f_shape[0], f_shape[1], f_shape[2], f_shape[3])
        };
        // bkkchw
        let yh = cols_shape[4];
        let yw = cols_shape[5];
        let batch_size = cols_shape[0];

        // Parameters for sgemm
        let num_elements_in_batch_y = ych * yh * yw;
        let num_elements_in_batch_c = {
            cols_shape[1] *
            cols_shape[2] *
            cols_shape[3] *
            cols_shape[4] *
            cols_shape[5]
        };
        let m = ych;
        let n = yh * yw;
        let k = xch * kh * kw;

        // Prepare buffers
        let c = unsafe {
            slice::from_raw_parts(cols.as_ptr(), cols.len())
        };
        let y = alloc_uninitialized_buf(batch_size * num_elements_in_batch_y);
        let w: &f32 = unsafe { &*w.as_ptr() };

        // Do task
        (0..batch_size).into_par_iter().for_each(|i| { // for each batch
            let c_region_head = &c[i * num_elements_in_batch_c];
            let y_region_head = &y[i * num_elements_in_batch_y];
            sgemm(false, false, w, c_region_head, y_region_head, m, n, k, 1., 0.);
        });

        // Move vectors into NdArrays
        let y = NdArray::from_shape_vec(
            ndarray::IxDyn(&[batch_size, ych, yh, yw]), y).unwrap();

        vec![Ok(y)]
    }

    fn grad(&self, gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>
    {
        let cols = xs[0];
        let w = xs[1];

        let gx = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                Conv2DTranspose {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        let gw = Tensor::builder()
            .set_inputs(vec![cols, gy, w])
            .build(
                Conv2DFilterGrad {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                }
            );

        vec![Some(gx), Some(gw)]
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
        cols: None,
    };

    let batch_size = 2;
    let xch = 2;
    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);
    let yh = get_yh!(&op, xh, kh);
    let yw = get_yw!(&op, xw, kw);
    let num_elements_in_batch_x = xch * xh * xw;
    let num_elements_in_batch_c = xch * kw * kh * yh * yw;
    let x = (0..(batch_size * num_elements_in_batch_x)).map(|a| a as f32).collect::<Vec<_>>();
    let c = alloc_uninitialized_buf(batch_size * num_elements_in_batch_c);
    // Call im2col on 2 chunks in parallel.
    (0..batch_size).into_par_iter().for_each(|i| { // for each mini-batch
        exec_im2col(
            &x[i * num_elements_in_batch_x],
            xch, xh, xw, kh, kw,
            op.pad_h, op.pad_w,
            op.stride_h, op.stride_w,
            op.dilation_h, op.dilation_w,
            &c[i * num_elements_in_batch_c]
        );
    });

    assert_eq!(
        c,
        vec![
            0.0, 1.0, 3.0, 4.0,
            1.0, 2.0, 4.0, 5.0,
            3.0, 4.0, 6.0, 7.0,
            4.0, 5.0, 7.0, 8.0,

            9.0, 10.0, 12.0, 13.0,
            10.0, 11.0, 13.0, 14.0,
            12.0, 13.0, 15.0, 16.0,
            13.0, 14.0, 16.0, 17.0,

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
        cols: None,
    };

    let xch = 2;
    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);
    let yh = get_yh!(&op, xh, kh);
    let yw = get_yw!(&op, xw, kw);

    let x = ndarray::Array1::range(0., (xch * xw * xh) as f32, 1.)
        .into_shape((1, xch as usize, xw as usize, xh as usize))
        .unwrap();

    let cols = alloc_uninitialized_buf(1 * xch * kw * kh * yh * yw);

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
fn test_conv2d()
{
    use ::op::Op;
    let op = Conv2D {
        pad_h: 0,
        pad_w: 0,
        stride_w: 1,
        stride_h: 1,
        dilation_h: 1,
        dilation_w: 1,
        cols: None,
    };

    let x = ndarray::Array1::range(0., 2.*2.*3.*3., 1.)
        .into_shape((2, 2, 3, 3)).unwrap().into_dyn();

    let w = ::ndarray_ext::ones(&[/*out_ch=*/2, /*in_ch=*/2, /*row=*/2, /*col=*/2]);

    let y = op.compute(::runtime::OpComputeContext {
        xs: vec![&x, &w],
        node: &::ops::zeros(&[0]) // dummy (not used)
    });

    assert_eq!(
        y[0].as_ref().unwrap().as_slice().unwrap(),
        &[52., 60., 76., 84., 52., 60., 76., 84.,
          196., 204., 220., 228., 196., 204., 220., 228.]
    );
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