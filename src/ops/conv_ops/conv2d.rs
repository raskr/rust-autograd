use NdArray;
use std::slice;
use super::*;

pub struct Conv2D {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
}

pub struct Conv2DFilterGrad {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

pub struct Conv2DWithCols {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
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
            let k_shape = w.shape();
            (k_shape[0], k_shape[2], k_shape[3])
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
            let x_region_head = &x[i * num_elements_in_batch_x];
            let c_region_head = &c[i * num_elements_in_batch_c];
            let y_region_head = &y[i * num_elements_in_batch_y];
            im2col(
                x_region_head,
                xch, xh, xw, kh, kw,
                self.pad_h, self.pad_w,
                self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w,
                c_region_head
            );
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
        let cols = &::ops::stop_gradient(::ops::nth_tensor(y, 1));

        let gx = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                super::conv2d_transpose::Conv2DTranspose {
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
            .set_inputs(vec![cols, gy, &::ops::stop_gradient(w)])
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
        let k_shape = w.shape();
        let (ych, xch, kh, kw) = {
            (k_shape[0], k_shape[1], k_shape[2], k_shape[3])
        };
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

    fn grad(&self, gy: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let cols = xs[0];
        let w = xs[1];

        let g_col = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                super::conv2d_transpose::Conv2DTranspose {
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

        vec![Some(g_col), Some(gw)]
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
        let k_shape = xs[2].shape();
        let cols_shape = cols.shape();
        let gy_shape = gy.shape();

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

        let (xch, kh, kw) = (k_shape[1], k_shape[2], k_shape[3]);
        // BUG: ych
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
        let gw = alloc_uninitialized_buf(ych * xch * kh * kw);
        let gw_head = unsafe { &*gw.as_ptr() };

        for i in 0..batch_size {
            sgemm(false, true,
                  &gy[i * num_elements_in_batch_g],
                  &cols[i * num_elements_in_batch_c],
                  gw_head, m, n, k, 1., (i != 0) as i32 as f32);
        }
        vec![Ok(NdArray::from_shape_vec(k_shape, gw).unwrap())]
    }

    fn grad(&self, ggw: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let cols = xs[0];
        let gy = xs[1];

        let gx = Tensor::builder()
            .set_inputs(vec![gy, ggw])
            .build(
                super::conv2d_transpose::Conv2DTranspose {
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
            .set_inputs(vec![cols, ggw])
            .build(
                Conv2DWithCols {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                }
            );

        vec![Some(gx), Some(ggy), None]
    }
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
        im2col(
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
        im2col_cpu(x.as_ptr(),
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