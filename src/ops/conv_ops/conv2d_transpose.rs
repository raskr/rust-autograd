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

struct Conv2DTransposeParams {
    batch_size: usize,
    xch: usize,
    xh: usize,
    xw: usize,
    ych: usize,
    yh: usize,
    yw: usize,
    kh: usize,
    kw: usize,
}

// Panics for invalid inputs
fn conv2d_transpose_extract_params<F: Float>(
    gy: &NdArrayView<F>,
    w: &NdArrayView<F>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<Conv2DTransposeParams, op::OpError> {
    if !crate::same_type::<F, f32>() && !crate::same_type::<F, f64>() {
        return Err(op::OpError::TypeUnsupported(
            "conv2d_transpose: only f32 and f64 are supported.".to_string(),
        ));
    }
    let gy_shape = gy.shape();
    let f_shape = w.shape();

    let batch_size = gy_shape[0];
    let ych = gy_shape[1];
    let yh = gy_shape[2];
    let yw = gy_shape[3];

    let xch = f_shape[1];
    let kh = f_shape[2];
    let kw = f_shape[3];
    let xh = stride_h * (yh - 1) - 2 * pad_h + (dilation_h * (kh - 1) + 1);
    let xw = stride_w * (yw - 1) - 2 * pad_w + (dilation_w * (kw - 1) + 1);

    if gy_shape.len() != 4 {
        return Err(op::OpError::IncompatibleShape(format!(
            "conv2d_transpose: Input must be 4D (got {:?})",
            gy_shape
        )));
    }
    if f_shape.len() != 4 {
        return Err(op::OpError::IncompatibleShape(format!(
            "conv2d_transpose: Filter must be 4D (got {:?})",
            f_shape
        )));
    }
    if ych != f_shape[0] {
        return Err(op::OpError::IncompatibleShape(format!(
            "conv2d_transpose: Number of input channels ({:?}) must match second filter dim ({:?})",
            ych, f_shape[0]
        )));
    }
    Ok(Conv2DTransposeParams {
        batch_size,
        xch,
        xh,
        xw,
        ych,
        yh,
        yw,
        kh,
        kw,
    })
}

fn conv2d_transpose_impl<F: Float>(
    gy: &NdArrayView<F>,
    w: &NdArrayView<F>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<NdArray<F>, op::OpError> {
    let Conv2DTransposeParams {
        batch_size,
        xch,
        xh,
        xw,
        ych,
        yh,
        yw,
        kh,
        kw,
    } = conv2d_transpose_extract_params(
        gy, w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    )?;

    // sgemm params
    let k = ych;
    let n = yh * yw;
    let m = kh * kw * xch;

    let size_per_batch_col = xch * kh * kw * yh * yw;

    let copied_gy = ndarray_ext::copy_if_not_standard(&gy);
    let copied_w = ndarray_ext::copy_if_not_standard(w);
    let gy_ptr = copied_gy.map(|inner| inner.as_ptr()).unwrap_or(gy.as_ptr());
    let w_ptr = copied_w.map(|inner| inner.as_ptr()).unwrap_or(w.as_ptr());
    let col_len = batch_size * size_per_batch_col;

    let mut col = Vec::with_capacity(col_len);
    #[cfg(not(feature = "mkl"))]
    {}

    let gx = unsafe {
        #[cfg(feature = "mkl")]
        {
            const GROUP_COUNT: usize = 1; // Fixed

            macro_rules! kernel_call_def {
                ($ty:ty, $f:ident) => {
                    if crate::same_type::<$ty, F>() {
                        $f(
                            CBLAS_ROW_MAJOR,
                            [CblasTrans; GROUP_COUNT].as_ptr(),
                            [CblasNoTrans; GROUP_COUNT].as_ptr(),
                            [m as MklInt; GROUP_COUNT].as_ptr(),
                            [n as MklInt; GROUP_COUNT].as_ptr(),
                            [k as MklInt; GROUP_COUNT].as_ptr(),
                            [1.; GROUP_COUNT].as_ptr(),
                            vec![w_ptr as *const _; batch_size].as_ptr(), // a array
                            [m as MklInt; GROUP_COUNT].as_ptr(),
                            get_batch_ptrs(batch_size, gy_ptr, gy.len()).as_ptr(), // b array
                            [n as MklInt; GROUP_COUNT].as_ptr(),
                            [0.; GROUP_COUNT].as_ptr(),
                            get_batch_ptrs_mut(batch_size, col.as_mut_ptr(), col_len)
                                .as_mut_ptr(), // c array
                            [n as MklInt; GROUP_COUNT].as_ptr(),
                            GROUP_COUNT as MklInt,
                            [batch_size as MklInt; GROUP_COUNT].as_ptr(),
                        );
                    }
                };
            }
            kernel_call_def!(f32, cblas_sgemm_batch);
            kernel_call_def!(f64, cblas_dgemm_batch);
            col.set_len(col_len);
        }

        #[cfg(not(feature = "mkl"))]
        {
            let w_slice = slice::from_raw_parts(w_ptr, w.len());
            let gy_slice = slice::from_raw_parts(gy_ptr, gy.len());
            let col_ref = {
                col.set_len(col_len);
                &col[0]
            };
            let size_per_batch_gy = ych * yh * yw;
            let (rsa, csa) = (1, m);
            let (rsb, csb) = (n, 1);
            let (rsc, csc) = (n, 1);
            macro_rules! kernel_call_def {
                ($ty:ty, $f:ident) => {
                    if same_type::<F, $ty>() {
                        (0..batch_size).into_par_iter().for_each(|i| {
                            let w = w_slice.as_ptr();
                            let gy_region_head = gy_slice.as_ptr().add(i * size_per_batch_gy);
                            let col_region_head = col_ref as *const F as *mut F;
                            let col_region_head = col_region_head.add(i * size_per_batch_col);
                            matrixmultiply::$f(
                                m,
                                k,
                                n,
                                1.,
                                w as *const _,
                                rsa as isize,
                                csa as isize,
                                gy_region_head as *const _,
                                rsb as isize,
                                csb as isize,
                                0.,
                                col_region_head as *mut _,
                                rsc as isize,
                                csc as isize,
                            );
                        });
                    }
                };
            }
            kernel_call_def!(f32, sgemm);
            kernel_call_def!(f64, dgemm);
        }

        col2im_batch(
            col.as_slice(),
            batch_size,
            xch as i32,
            xh as i32,
            xw as i32,
            kh as i32,
            kw as i32,
            pad_h as i32,
            pad_w as i32,
            stride_h as i32,
            stride_w as i32,
            dilation_h as i32,
            dilation_w as i32,
        )
    };
    // return gx
    unsafe {
        Ok(NdArray::from_shape_vec_unchecked(
            ndarray::IxDyn(&[batch_size, xch, xh, xw]),
            gx,
        ))
    }
}

impl<T: Float> crate::op::Op<T> for Conv2DTranspose {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let gy = &ctx.input(0); // (batch, ych, yh, yw)
        let w = &ctx.input(1); // (ych, xch, kh, kw)
        let gx = conv2d_transpose_impl(
            gy,
            w,
            self.pad,
            self.pad,
            self.stride,
            self.stride,
            self.dilation,
            self.dilation,
        );
        match gx {
            Ok(gx) => {
                ctx.append_output(gx);
            }
            Err(e) => {
                ctx.set_error(e);
            }
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let x = ctx.input(0);
        let w = ctx.input(1);
        let gy = ctx.output_grad();

        let gx = Tensor::builder().set_ro_inputs(&[&gy, &w]).build(
            s,
            super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let gw = Tensor::builder()
            .set_ro_inputs(&[&gy, &x, &s.stop_gradient(w)])
            .build(
                s,
                Conv2DTransposeFilterGrad {
                    pad: self.pad,
                    stride: self.stride,
                    dilation: self.dilation,
                },
            );

        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(Some(gw));
    }
}

fn conv2d_transpose_filter_grad_impl<F: Float>(
    x: &NdArrayView<F>,
    w: &NdArrayView<F>,
    gy: &NdArrayView<F>,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> NdArray<F> {
    let k_shape = w.shape();
    let x_shape = x.shape();
    let gy_shape = gy.shape();

    let batch_size = x_shape[0];
    let (kh, kw) = (k_shape[2], k_shape[3]);
    let (xh, xw) = (gy_shape[2], gy_shape[3]);
    let yh = (xh + 2 * pad_h - (dilation_h * (kh - 1) + 1)) / stride_h + 1;
    let yw = (xw + 2 * pad_w - (dilation_w * (kw - 1) + 1)) / stride_w + 1;
    let (ych, xch) = (x_shape[1], gy_shape[1]);
    let size_per_batch_cols = yh * yh * kh * kw * gy_shape[1];
    let size_per_batch_x = x_shape[1] * x_shape[2] * x_shape[3];

    let x = unsafe { slice::from_raw_parts(x.as_ptr(), x.len()) };
    let gy = unsafe { slice::from_raw_parts(gy.as_ptr(), gy.len()) };

    // gy -> gy_cols
    let gy_cols = im2col_batch(
        gy,
        batch_size as usize,
        gy_shape[1] as i32,
        gy_shape[2] as i32,
        gy_shape[3] as i32,
        kh as i32,
        kw as i32,
        pad_h as i32,
        pad_w as i32,
        stride_h as i32,
        stride_w as i32,
        dilation_h as i32,
        dilation_w as i32,
    );

    let gw_size = k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3];
    let mut gw = Vec::with_capacity(gw_size);
    let gw_head = gw.as_mut_ptr();

    #[cfg(feature = "mkl")]
    {
        // sgemm params
        let m = ych;
        let n = kh * kw * xch;
        let k = yh * yw;

        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                unsafe {
                    if crate::same_type::<$ty, F>() {
                        for i in 0..batch_size {
                            let x_region_head = &x[i * size_per_batch_x] as *const F;
                            let gy_col_region_ptr = &gy_cols[i * size_per_batch_cols] as *const F;
                            $f(
                                CBLAS_ROW_MAJOR,
                                CblasNoTrans,
                                CblasTrans,
                                m as MklInt,
                                n as MklInt,
                                k as MklInt,
                                1.,
                                x_region_head as *const $ty,
                                k as MklInt, // a array
                                gy_col_region_ptr as *const $ty,
                                k as MklInt, // b array
                                if i == 0 { 0. } else { 1. },
                                gw_head as *mut $ty, // c array
                                n as MklInt,
                            );
                        }
                    }
                }
            };
        }
        kernel_call_def!(f32, cblas_sgemm);
        kernel_call_def!(f64, cblas_dgemm);
    }

    #[cfg(not(feature = "mkl"))]
    {
        let (m, n, k) = (ych, kh * kw * xch, yh * yw);
        let (rsa, csa) = (k, 1);
        let (rsb, csb) = (1, k);
        let (rsc, csc) = (n, 1);
        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                unsafe {
                    if crate::same_type::<$ty, F>() {
                        for i in 0..batch_size {
                            let x_region_head = &x[i * size_per_batch_x] as *const F;
                            let gy_col_region_ptr = &gy_cols[i * size_per_batch_cols] as *const F;
                            matrixmultiply::$f(
                                m,
                                k,
                                n,
                                1., // alpha
                                x_region_head as *const $ty,
                                rsa as isize,
                                csa as isize,
                                gy_col_region_ptr as *const $ty,
                                rsb as isize,
                                csb as isize,
                                if i == 0 { 0. } else { 1. }, // beta
                                gw_head as *mut $ty,          // c
                                rsc as isize,
                                csc as isize,
                            );
                        }
                    }
                }
            };
        }
        kernel_call_def!(f32, sgemm);
        kernel_call_def!(f64, dgemm);
    }
    unsafe {
        gw.set_len(gw_size);
        NdArray::from_shape_vec_unchecked(k_shape, gw)
    }
}

impl<T: Float> crate::op::Op<T> for Conv2DTransposeFilterGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let gy = &ctx.input(0);
        let x = &ctx.input(1);
        let w = &ctx.input(2);
        let gw = conv2d_transpose_filter_grad_impl(
            x,
            w,
            gy,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.stride,
            self.stride,
        );
        ctx.append_output(gw);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let gy = ctx.input(0);
        let gw = ctx.output_grad();
        let x = ctx.input(1);

        let ggy = Tensor::builder().set_ro_inputs(&[&x, &gw]).build(
            s,
            Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let ggx = Tensor::builder().set_ro_inputs(&[&gy, &gw]).build(
            s,
            super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        ctx.append_input_grad(Some(ggy));
        ctx.append_input_grad(Some(ggx));
        ctx.append_input_grad(None);
    }
}

#[test]
fn test_deconv() {
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;
    let ans = NdArray::<f32>::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3, 3]),
        vec![
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0,
            2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0,
            4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0,
            2.0, 4.0, 2.0,
        ],
    )
    .unwrap();

    let out_val = crate::run(|s: &mut crate::Graph<f32>| {
        let w = s.ones(&[ych, xch, kh, kw]);
        let g = s.ones(&[batch_size, ych, yh, yw]);
        let out = s.conv2d_transpose(g, w, 0, 1);
        out.eval(&[]).unwrap()
    });
    out_val.all_close(&ans, 1e-3);
}
