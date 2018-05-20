extern crate ndarray;
extern crate libc;
extern crate rayon;

use self::rayon::iter::*;
use self::libc::{c_float, c_int};
use ndarray_ext::NdArray;
use tensor::Tensor;

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

    #[allow(dead_code)]
    fn co2im_kernel(
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
        data_im: *mut c_float,
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

pub struct Conv2D {
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

#[inline]
fn get_col_w(op: &Conv2D, xw: i32, kw: i32) -> i32
{
    (xw + 2 * op.pad_w - (op.dilation_w * (kw - 1) + 1)) / op.stride_w + 1
}

#[inline]
fn get_col_h(op: &Conv2D, xh: i32, kh: i32) -> i32
{
    (xh + 2 * op.pad_h - (op.dilation_h * (kh - 1) + 1)) / op.stride_h + 1
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

        let yh = get_col_h(self, xh, kh);
        let yw = get_col_w(self, xw, kw);

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
            type Mat = ndarray::Array2<f32>;
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
    let yh = get_col_h(&op, xh, kh);
    let yw = get_col_w(&op, xw, kw);
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
    let yh = get_col_h(&op, xh, kh);
    let yw = get_col_w(&op, xw, kw);

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
    use ::op::Op;

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