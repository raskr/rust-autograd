use super::*;

pub struct Conv2DTranspose {
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub cols: Option<Vec<f32>>
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

        // Alloc buffers as necessary
        let col = get_or_insert_cols!(self, batch_size, num_elements_in_batch_col);
        // Col2im buffer must be initialized with zeros
        let gx = vec![0.; batch_size * num_elements_in_batch_gx];

        (0..batch_size).into_par_iter().for_each(|i| { // for each mini-batch
            let gy_region_head = &gy[i * num_elements_in_batch_gy];
            let col_region_head = &col[i * num_elements_in_batch_col];
            let gx_region_head = &gx[i * num_elements_in_batch_gx];
            sgemm(true, false, w, gy_region_head, col_region_head, m, n, k, 1., 0.);
            col2im(col_region_head, xch, xh, xw, kh, kw,
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

        let gx = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(
                super::conv2d::Conv2D {
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
            .set_inputs(vec![gy, x, &w])
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

        vec![Some(gx), Some(gw)]
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
        let k_shape = xs[2].shape();

        let x_shape = x.shape();
        let gy_shape = gy.shape();

        let batch_size = x_shape[0];
        let (kh, kw) = (k_shape[2], k_shape[3]);

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

        let gw = alloc_uninitialized_buf(k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3]);
        let gw_head = unsafe { &*gw.as_ptr() };

        for i in 0..batch_size {
            let x_region_head = &x[i * num_elements_in_batch_x];
            let c_region_head = &cols[i * num_elements_in_batch_c];
            let g_region_head = &gy[i * num_elements_in_batch_g];
            im2col(
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

        vec![Ok(NdArray::from_shape_vec(k_shape, gw).unwrap())]
    }

    fn grad(&self, gw: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gy = xs[0];
        let x = xs[1];

        let ggy = Tensor::builder()
            .set_inputs(vec![x, gw])
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

        let ggx = Tensor::builder()
            .set_inputs(vec![gy, gw])
            .build(
                super::conv2d::Conv2D {
                    pad_h: self.pad_h,
                    pad_w: self.pad_w,
                    stride_h: self.stride_h,
                    stride_w: self.stride_w,
                    dilation_h: self.dilation_h,
                    dilation_w: self.dilation_w,
                    cols: None,
                }
            );

        vec![Some(ggy), Some(ggx), None]
    }
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
            col2im_cpu(cols_head,
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
