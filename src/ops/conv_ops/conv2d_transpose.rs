use super::*;

pub struct Conv2DTranspose {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

pub struct Conv2DTransposeFilterGrad {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl<T: Float> ::op::Op<T> for Conv2DTranspose {
    fn name(&self) -> &str {
        "Conv2DTranspose"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();

        let gy: &NdArray<T> = xs[0]; // (batch, ych, yh, yw)
        let w: &NdArray<T> = xs[1]; // (ych, xch, kh, kw)
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

        assert_eq!(
            gy_shape.len(),
            4,
            "ag::conv2d: Input must be 4D (got {:?})",
            gy_shape
        );
        assert_eq!(
            f_shape.len(),
            4,
            "ag::conv2d: Filter must be 4D (got {:?})",
            f_shape
        );
        assert_eq!(
            ych, f_shape[0],
            "ag::conv2d: Number of input channels ({:?}) must match second filter dim ({:?})",
            ych, f_shape[0]
        );

        // sgemm params
        let k = ych;
        let n = yh * yw;
        let m = kh * kw * xch;

        let size_per_batch_col = xch * kh * kw * yh * yw;

        let gy = unsafe { slice::from_raw_parts(gy.as_ptr(), gy.len()) };
        let col = ::dot_ops::uninitialized_vec(batch_size * size_per_batch_col);
        // Col2im buffer must be initialized with zeros

        #[cfg(feature = "mkl")]
        {
            let w = w.as_ptr();
            if same_type::<T, f32>() {
                ::ops::dot_ops::cblas_sgemm_batch_wrapper(
                    true,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f32; batch_size],
                    ::dot_ops::get_region_heads(batch_size, gy),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, col.as_slice()),
                    1,
                    batch_size,
                );
            } else if same_type::<T, f64>() {
                ::ops::dot_ops::cblas_dgemm_batch_wrapper(
                    true,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f64; batch_size],
                    ::dot_ops::get_region_heads(batch_size, gy),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, col.as_slice()),
                    1,
                    batch_size,
                );
            } else {
                panic!("gemm supports only f32 and f64.")
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            let size_per_batch_gy = ych * yh * yw;
            (0..batch_size).into_par_iter().for_each(|i| unsafe {
                let w: *const T = mem::transmute(w.as_ptr());
                let gy_region_head = &gy[i * size_per_batch_gy] as *const T;
                let col_region_head: *mut T = mem::transmute(&col[i * size_per_batch_col]);
                slow_gemm!(
                    true,
                    false,
                    w as *const f32,
                    gy_region_head as *const f32,
                    col_region_head as *mut f32,
                    m,
                    n,
                    k,
                    1.,
                    0.
                );
            });
        }

        let gx = col2im_batch(
            col.as_slice(),
            batch_size,
            xch as isize,
            xh as isize,
            xw as isize,
            kh as isize,
            kw as isize,
            self.pad as isize,
            self.pad as isize,
            self.stride as isize,
            self.stride as isize,
            self.dilation as isize,
            self.dilation as isize,
        );

        let gx = NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, xch, xh, xw]), gx);
        vec![Ok(gx.unwrap())]
    }

    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = xs[0];
        let w = xs[1];

        let gx = Tensor::builder()
            .set_inputs(vec![gy, w])
            .build(super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let gw = Tensor::builder()
            .set_inputs(vec![gy, x, &::ops::stop_gradient(w)])
            .build(Conv2DTransposeFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        vec![Some(gx), Some(gw)]
    }
}

impl<T: Float> ::op::Op<T> for Conv2DTransposeFilterGrad {
    fn name(&self) -> &str {
        "Conv2DTransposeFilterGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let gy = xs[0];
        let x = xs[1];
        let k_shape = xs[2].shape();

        let x_shape = x.shape();
        let gy_shape = gy.shape();

        let batch_size = x_shape[0];
        let (kh, kw) = (k_shape[2], k_shape[3]);

        let size_per_batch_c = {
            get_yh!(self, gy_shape[2], kh) * get_yw!(self, gy_shape[3], kw) * kh * kw * gy_shape[1]
        };
        let size_per_batch_x = x_shape[1] * x_shape[2] * x_shape[3];

        // sgemm params
        let m = x_shape[1];
        let n = kh * kw * gy_shape[1];
        let k = get_yh!(self, gy_shape[2], kh) * get_yw!(self, gy_shape[3], kw);

        let x = unsafe { slice::from_raw_parts(x.as_ptr(), x.len()) };
        let gy = unsafe { slice::from_raw_parts(gy.as_ptr(), gy.len()) };
        let mut gw = uninitialized_vec(k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3]);
        let gw_head = gw.as_mut_ptr();

        let cols = im2col_batch(
            gy,
            batch_size,
            gy_shape[1] as isize,
            gy_shape[2] as isize,
            gy_shape[3] as isize,
            kh as isize,
            kw as isize,
            self.pad as isize,
            self.pad as isize,
            self.stride as isize,
            self.stride as isize,
            self.dilation as isize,
            self.dilation as isize,
        );

        for i in 0..batch_size {
            let x_region_head = &x[i * size_per_batch_x] as *const T;
            let c_region_head = &cols[i * size_per_batch_c] as *const T;
            #[cfg(feature = "mkl")]
            {
                if same_type::<T, f32>() {
                    ::ops::dot_ops::cblas_sgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        x_region_head as *const f32,
                        c_region_head as *const f32,
                        if i == 1 { 1. } else { 0. },
                        gw_head as *mut f32,
                    )
                } else if same_type::<T, f64>() {
                    ::ops::dot_ops::cblas_dgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        x_region_head as *const f64,
                        c_region_head as *const f64,
                        if i == 1 { 1. } else { 0. },
                        gw_head as *mut f64,
                    )
                }
            }
            #[cfg(not(feature = "mkl"))]
            {
                unsafe {
                    slow_gemm!(
                        false,
                        true,
                        x_region_head as *const f32,
                        c_region_head as *const f32,
                        gw_head as *mut f32,
                        m,
                        n,
                        k,
                        1.,
                        if i == 1 { 1. } else { 0. }
                    );
                }
            }
        }

        vec![Ok(NdArray::from_shape_vec(k_shape, gw).unwrap())]
    }

    fn grad(&self, gw: &Tensor<T>, xs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gy = xs[0];
        let x = xs[1];

        let ggy = Tensor::builder()
            .set_inputs(vec![x, gw])
            .build(Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let ggx = Tensor::builder()
            .set_inputs(vec![gy, gw])
            .build(super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        vec![Some(ggy), Some(ggx), None]
    }
}

#[test]
fn test_tensor_size_after_convolution_t() {
    let op = Conv2DTranspose {
        pad: 0,
        stride: 1,
        dilation: 1,
    };
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh!(&op, yh, kh);
    let xw = get_xw!(&op, yw, kw);
    assert_eq!(xh, 3);
    assert_eq!(xw, 3);
}

#[test]
fn test_deconv() {
    use op::Op;
    let op = Conv2DTranspose {
        pad: 0,
        stride: 1,
        dilation: 1,
    };
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let (xh, xw) = (3, 3);
    let batch_size = 2;

    let w = ::ndarray_ext::ones::<f32>(&[ych, xch, kh, kw]);
    let g = ::ndarray_ext::ones(&[batch_size, ych, yh, yw]);

    let ret = op.compute(::runtime::OpComputeContext::new(
        &::ops::zeros(&[0]), // dummy (not used)
        vec![&g, &w],
    ));

    let x = ::ndarray_ext::ones::<f32>(&[batch_size, xch, xh, xw]);
    assert_eq!(x.shape(), ret[0].as_ref().unwrap().shape());

    assert_eq!(
        ret[0].clone().unwrap().into_raw_vec(),
        vec![
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0,
            2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0,
            4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0,
            2.0, 4.0, 2.0,
        ]
    )
}
