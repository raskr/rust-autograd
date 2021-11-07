use super::*;
use crate::tensor_ops::*;

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

    let copied_gy;
    let copied_w;
    let gy_slice;
    let w_slice;

    if let Some(gy) = gy.as_slice() {
        gy_slice = gy;
    } else {
        copied_gy = ndarray_ext::deep_copy(gy);
        unsafe {
            gy_slice = slice::from_raw_parts(copied_gy.as_ptr(), copied_gy.len());
        }
    }

    if let Some(w) = w.as_slice() {
        w_slice = w;
    } else {
        copied_w = ndarray_ext::deep_copy(w);
        unsafe {
            w_slice = slice::from_raw_parts(copied_w.as_ptr(), copied_w.len());
        }
    }

    let gy_size_per_batch = gy.len() / batch_size;
    let cols_size_per_batch = xch * kh * kw * yh * yw;
    let ret_size_per_batch = (xch * xh * xw) as usize;

    let cols_size = batch_size * cols_size_per_batch;
    let ret_size = batch_size * ret_size_per_batch;

    let mut cols: Vec<F> = Vec::with_capacity(cols_size);
    unsafe {
        cols.set_len(cols_size);
    }
    let mut ret = vec![F::zero(); ret_size]; // = gx

    let a = gy_slice.par_iter().step_by(gy_size_per_batch);
    let b = cols.par_iter_mut().step_by(cols_size_per_batch);
    let c = ret.par_iter_mut().step_by(ret_size_per_batch);

    let gx = unsafe {
        a.zip_eq(b)
            .zip_eq(c)
            .for_each(move |((gy_ptr, cols), ret)| {
                #[cfg(feature = "blas")]
                {
                    macro_rules! kernel_call_def {
                        ($ty:ty, $f:ident) => {
                            if crate::same_type::<$ty, F>() {
                                $f(
                                    CblasRowMajor,
                                    CblasTrans,
                                    CblasNoTrans,
                                    m as BlasIF,                    // m, rows of Op(a)
                                    n as BlasIF,                    // n, cols of Op(b)
                                    k as BlasIF,                    // k, cols of Op(a)
                                    1.,                             // alpha
                                    w_slice.as_ptr() as *const _,   // a
                                    m as BlasIF,                    // lda
                                    gy_ptr as *const F as *const _, // b
                                    n as BlasIF,                    // ldb
                                    0.,                             // beta
                                    cols as *mut F as *mut _,       // c
                                    n as BlasIF,                    // ldc
                                );
                            }
                        };
                    }
                    kernel_call_def!(f32, cblas_sgemm);
                    kernel_call_def!(f64, cblas_dgemm);
                }

                #[cfg(not(feature = "blas"))]
                {
                    let (rsa, csa) = (1, m);
                    let (rsb, csb) = (n, 1);
                    let (rsc, csc) = (n, 1);
                    macro_rules! kernel_call_def {
                        ($ty:ty, $f:ident) => {
                            if same_type::<F, $ty>() {
                                matrixmultiply::$f(
                                    m,
                                    k,
                                    n,
                                    1.,
                                    w_slice.as_ptr() as *const _,
                                    rsa as isize,
                                    csa as isize,
                                    gy_ptr as *const F as *const _,
                                    rsb as isize,
                                    csb as isize,
                                    0.,
                                    cols as *mut F as *mut _,
                                    rsc as isize,
                                    csc as isize,
                                );
                            }
                        };
                    }
                    kernel_call_def!(f32, sgemm);
                    kernel_call_def!(f64, dgemm);
                }

                col2im(
                    cols as *const _,
                    ret as *mut _,
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
                );
            });
        ret
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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
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
                Ok(())
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let w = ctx.input(1);
        let gy = ctx.output_grad();

        let gx = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&w, false)
            .build(super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let gw = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&x, false)
            .append_input(&stop_gradient(w), false)
            .build(Conv2DTransposeFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

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

    #[cfg(feature = "blas")]
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
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasTrans,
                                m as BlasIF,
                                n as BlasIF,
                                k as BlasIF,
                                1.,
                                x_region_head as *const $ty,
                                k as BlasIF, // a array
                                gy_col_region_ptr as *const $ty,
                                k as BlasIF, // b array
                                if i == 0 { 0. } else { 1. },
                                gw_head as *mut $ty, // c array
                                n as BlasIF,
                            );
                        }
                    }
                }
            };
        }
        kernel_call_def!(f32, cblas_sgemm);
        kernel_call_def!(f64, cblas_dgemm);
    }

    #[cfg(not(feature = "blas"))]
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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
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
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.input(0);
        let gw = ctx.output_grad();
        let x = ctx.input(1);

        let ggy = Tensor::builder(ctx.graph())
            .append_input(&x, false)
            .append_input(&gw, false)
            .build(Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let ggx = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&gw, false)
            .build(super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        ctx.append_input_grad(Some(ggy));
        ctx.append_input_grad(Some(ggx));
        ctx.append_input_grad(None);
    }
}

#[test]
fn test_deconv() {
    use crate::tensor_ops as T;

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

    let mut ctx = crate::VariableEnvironment::new();
    let out_val = ctx.run(|graph| {
        let w = T::ones(&[ych, xch, kh, kw], graph);
        let g = T::ones(&[batch_size, ych, yh, yw], graph);
        let out = T::conv2d_transpose(g, w, 0, 1);
        out.eval(graph).unwrap()
    });
    out_val.all_close(&ans, 1e-3);
}
