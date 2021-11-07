use super::*;
use crate::tensor_ops::*;
use ndarray::IxDyn;
use std::slice;

pub struct Conv2D {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

pub struct Conv2DFilterGrad {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

pub struct Conv2DWithCols {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[cfg(feature = "blas")]
fn fast_im2col_gemm_fused_kernel<F: Float>(
    x: &[F], // 4-dimensional
    filter: &[F],
    batch_size: usize, // x.shape[0]
    xch: i32,          // number of channels of x
    ych: i32,          // number of channels of x
    xh: i32,
    xw: i32, // x (input) height, width
    kh: i32,
    kw: i32, // kernel height, width
    ph: i32,
    pw: i32, // padding height, width
    sh: i32,
    sw: i32, // stride height, width
    dh: i32,
    dw: i32, // dilation height, width
) -> (Vec<F>, Vec<F>) {
    let yh = (xh + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    let yw = (xw + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    let channel_size = (xh * xw) as usize;

    let col_size_per_batch = (xch * kw * kh * yh * yw) as usize;
    let y_size_per_batch = (ych * yh * yw) as usize;

    let mut y = Vec::with_capacity(batch_size * y_size_per_batch);
    let mut cols = Vec::with_capacity(batch_size * col_size_per_batch);
    unsafe {
        y.set_len(batch_size * y_size_per_batch);
        cols.set_len(batch_size * col_size_per_batch)
    }

    let a = cols.par_iter_mut().step_by(col_size_per_batch);
    let b = x.par_iter().step_by(xch as usize * channel_size);
    let c = y.par_iter_mut().step_by(y_size_per_batch);

    // Parallelize im2col + gemm
    a.zip_eq(b).zip_eq(c).for_each(move |((cols, x), y)| {
        unsafe {
            // mutate cols
            im2col(
                x as *const _,
                cols as *mut _,
                xch,
                xh,
                xw,
                kh,
                kw,
                ph,
                pw,
                sh,
                sw,
                dh,
                dw,
            );

            let m = ych as BlasIF;
            let n = (yh * yw) as BlasIF;
            let k = (xch * kh * kw) as BlasIF;
            macro_rules! kernel_call_def {
                ($ty:ty, $f:ident) => {
                    // invalid type must be reported beforehand
                    if crate::same_type::<$ty, F>() {
                        $f(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            m as BlasIF,                  // m, rows of Op(a)
                            n as BlasIF,                  // n, cols of Op(b)
                            k as BlasIF,                  // k, cols of Op(a)
                            1.,                           // alpha
                            filter.as_ptr() as *const _,  // a
                            k,                            // lda
                            cols as *const F as *const _, // b
                            n,                            // ldb
                            0.,                           // beta
                            y as *mut F as *mut $ty,      // c
                            n,                            // ldc
                        );
                    }
                };
            }
            kernel_call_def!(f32, cblas_sgemm);
            kernel_call_def!(f64, cblas_dgemm);
        }
    });

    (y, cols)
}

#[cfg(not(feature = "blas"))]
fn slow_im2col_gemm_fused_kernel<F: Float>(
    x: &[F], // 4-dimensional
    filter: &[F],
    batch_size: usize, // x.shape[0]
    xch: i32,          // number of channels of x
    ych: i32,          // number of channels of y
    xh: i32,
    xw: i32, // x (input) height, width
    kh: i32,
    kw: i32, // kernel height, width
    ph: i32,
    pw: i32, // padding height, width
    sh: i32,
    sw: i32, // stride height, width
    dh: i32,
    dw: i32, // dilation height, width
) -> (Vec<F>, Vec<F>) {
    let yh = (xh + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    let yw = (xw + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    let channel_size = (xh * xw) as usize;

    let m = ych;
    let n = yh * yw;
    let k = xch * kh * kw;

    let col_size_per_batch = (xch * kw * kh * yh * yw) as usize;
    let y_size_per_batch = (ych * yh * yw) as usize;

    let mut y = Vec::with_capacity(batch_size * y_size_per_batch);
    let mut cols = Vec::with_capacity(batch_size * col_size_per_batch);
    unsafe {
        y.set_len(batch_size * y_size_per_batch);
        cols.set_len(batch_size * col_size_per_batch)
    }

    let a = y.par_iter_mut().step_by(y_size_per_batch);
    let b = cols.par_iter_mut().step_by(col_size_per_batch);
    let c = x.par_iter().step_by(xch as usize * channel_size);

    a.zip_eq(b).zip_eq(c).for_each(move |((y, cols), x)| {
        // mutate cols
        im2col(
            x as *const _,
            cols as *mut _,
            xch,
            xh,
            xw,
            kh,
            kw,
            ph,
            pw,
            sh,
            sw,
            dh,
            dw,
        );

        // Col ok

        // let mut y = &mut y[i * y_size_per_batch.. (i+1) * y_size_per_batch];
        let (rsa, csa) = (k, 1);
        let (rsb, csb) = (n, 1);
        let (rsc, csc) = (n, 1);
        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<$ty, F>() {
                    unsafe {
                        matrixmultiply::$f(
                            m as usize,
                            k as usize,
                            n as usize,
                            1.,
                            filter.as_ptr() as *const $ty,
                            rsa as isize,
                            csa as isize,
                            cols as *const F as *const $ty,
                            rsb as isize,
                            csb as isize,
                            0.,
                            y as *mut F as *mut $ty,
                            rsc as isize,
                            csc as isize,
                        );
                    }
                }
            };
        }
        kernel_call_def!(f32, sgemm);
        kernel_call_def!(f64, dgemm);
    });

    unsafe {
        y.set_len(batch_size * y_size_per_batch);
        cols.set_len(batch_size * col_size_per_batch);
        (y, cols)
    }
}

#[cfg(feature = "blas")]
// inputs must be row-major matrices
fn fast_col_x_filter_kernel<F: Float>(
    cols: &[F],
    filter: &[F],
    xch: usize,
    ych: usize,
    yh: usize,
    yw: usize,
    kh: usize,
    kw: usize,
    batch_size: usize,
) -> Vec<F> {
    let y_len = batch_size * ych * yh * yw;
    let mut y = Vec::with_capacity(y_len);
    unsafe {
        y.set_len(y_len);
    }
    // params for blas gemm
    let m = ych as BlasIF;
    let n = (yh * yw) as BlasIF;
    let k = (xch * kh * kw) as BlasIF;
    let col_size_per_batch = (xch * kw * kh * yh * yw) as usize;
    let y_size_per_batch = (ych * yh * yw) as usize;

    let a = y.par_iter_mut().step_by(y_size_per_batch);
    let b = cols.par_iter().step_by(col_size_per_batch);

    a.zip_eq(b).for_each(move |(y, cols)| {
        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<$ty, F>() {
                    unsafe {
                        $f(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            m as BlasIF,                  // m, rows of Op(a)
                            n as BlasIF,                  // n, cols of Op(b)
                            k as BlasIF,                  // k, cols of Op(a)
                            1.,                           // alpha
                            filter.as_ptr() as *const _,  // a
                            k,                            // lda
                            cols as *const F as *const _, // b
                            n,                            // ldb
                            0.,                           // beta
                            y as *mut F as *mut _,        // c
                            n,                            // ldc
                        );
                    }
                }
            };
        }
        kernel_call_def!(f32, cblas_sgemm);
        kernel_call_def!(f64, cblas_dgemm);
    });

    unsafe {
        y.set_len(y_len);
    }
    y
}

#[cfg(not(feature = "blas"))]
fn slow_col_x_filter_kernel<F: Float>(
    cols: &[F],
    filter: &[F],
    xch: usize,
    ych: usize,
    yh: usize,
    yw: usize,
    kh: usize,
    kw: usize,
    batch_size: usize,
) -> Vec<F> {
    let size_per_batch_y = ych * yh * yw;
    let mut y = Vec::with_capacity(batch_size * size_per_batch_y);
    unsafe {
        y.set_len(batch_size * size_per_batch_y);
    }
    let m = ych;
    let n = yh * yw;
    let k = xch * kh * kw;
    let (rsa, csa) = (k, 1);
    let (rsb, csb) = (n, 1);
    let (rsc, csc) = (n, 1);
    let size_per_batch_cols = xch * kw * kh * yh * yw;

    macro_rules! kernel_call_def {
        ($ty:ty, $f:ident) => {
            if crate::same_type::<$ty, F>() {
                let a = cols.par_iter().step_by(size_per_batch_cols);
                let b = y.par_iter_mut().step_by(size_per_batch_y);

                a.zip_eq(b).for_each(move |(cols, y)| unsafe {
                    matrixmultiply::$f(
                        m,
                        k,
                        n,
                        1.,
                        filter.as_ptr() as *const $ty,
                        rsa as isize,
                        csa as isize,
                        cols as *const F as *const $ty,
                        rsb as isize,
                        csb as isize,
                        0.,
                        y as *mut F as *mut $ty,
                        rsc as isize,
                        csc as isize,
                    );
                });
            }
        };
    }
    kernel_call_def!(f32, sgemm);
    kernel_call_def!(f64, dgemm);
    unsafe {
        y.set_len(batch_size * size_per_batch_y);
    }
    y
}

struct Conv2DParams {
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
fn conv2d_extract_params<F: Float>(
    x: &NdArrayView<F>,
    w: &NdArrayView<F>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<Conv2DParams, op::OpError> {
    if !crate::same_type::<F, f32>() && !crate::same_type::<F, f64>() {
        return Err(op::OpError::TypeUnsupported(
            "conv2d: only f32 and f64 are supported.".to_string(),
        ));
    }
    // Extract size params
    let (batch_size, xch, xh, xw) = {
        let x_shape = x.shape();
        if x_shape.len() != 4 {
            return Err(op::OpError::IncompatibleShape(format!(
                "conv2d: lhs input must be 4D (got {:?})",
                x_shape
            )));
        }
        (x_shape[0], x_shape[1], x_shape[2], x_shape[3])
    };
    let (ych, kh, kw) = {
        let w_shape = w.shape();
        if w_shape.len() != 4 {
            return Err(op::OpError::IncompatibleShape(format!(
                "conv2d: filter must be 4D (got {:?})",
                w_shape
            )));
        }
        if xch != w_shape[1] {
            return Err(op::OpError::IncompatibleShape(format!(
                "conv2d: input channel dim ({:?}) must match filter's second dim ({:?})",
                xch, w_shape[1]
            )));
        }
        (w_shape[0], w_shape[2], w_shape[3])
    };
    let yh = (xh + 2 * pad_h - (dilation_h * (kh - 1) + 1)) / stride_h + 1;
    let yw = (xw + 2 * pad_w - (dilation_w * (kw - 1) + 1)) / stride_w + 1;
    Ok(Conv2DParams {
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

/// Returns: (conv result, im2col result)
fn conv2d_impl<F: Float>(
    x: &NdArrayView<F>,
    w: &NdArrayView<F>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<(NdArray<F>, NdArray<F>), op::OpError> {
    let Conv2DParams {
        batch_size,
        xch,
        xh,
        xw,
        ych,
        yh,
        yw,
        kh,
        kw,
    } = conv2d_extract_params(
        x, w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    )?;

    let copied_x;
    let copied_w;
    let x_slice;
    let w_slice;

    if let Some(x) = x.as_slice() {
        x_slice = x;
    } else {
        copied_x = ndarray_ext::deep_copy(x);
        unsafe {
            x_slice = slice::from_raw_parts(copied_x.as_ptr(), copied_x.len());
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

    unsafe {
        let f;
        #[cfg(feature = "blas")]
        {
            f = fast_im2col_gemm_fused_kernel;
        }
        #[cfg(not(feature = "blas"))]
        {
            f = slow_im2col_gemm_fused_kernel;
        }

        let (y, cols) = f(
            x_slice,
            w_slice,
            batch_size,
            xch as i32,
            ych as i32,
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
        let y = NdArray::from_shape_vec(IxDyn(&[batch_size, ych, yh, yw]), y).unwrap();
        let cols =
            NdArray::from_shape_vec_unchecked(IxDyn(&[batch_size, xch, kw, kh, yh, yw]), cols);
        Ok((y, cols))
    }
}

fn conv2d_with_cols_impl<F: Float>(cols: &NdArrayView<F>, w: &NdArrayView<F>) -> NdArray<F> {
    // Extract size params
    let cols_shape = cols.shape();
    let k_shape = w.shape();
    let (ych, xch, kh, kw) = { (k_shape[0], k_shape[1], k_shape[2], k_shape[3]) };
    let (yh, yw) = (cols_shape[4], cols_shape[5]);
    let batch_size = cols_shape[0];

    let copied_w;
    let w_slice;
    if let Some(w) = w.as_slice() {
        w_slice = w;
    } else {
        copied_w = ndarray_ext::deep_copy(w);
        unsafe {
            w_slice = slice::from_raw_parts(copied_w.as_ptr(), copied_w.len());
        }
    }

    let f;
    #[cfg(feature = "blas")]
    {
        f = fast_col_x_filter_kernel;
    }
    #[cfg(not(feature = "blas"))]
    {
        f = slow_col_x_filter_kernel;
    }
    let y = f(
        cols.as_slice().unwrap(),
        w_slice,
        xch,
        ych,
        yh,
        yw,
        kh,
        kw,
        batch_size,
    );
    unsafe { NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[batch_size, ych, yh, yw]), y) }
}

impl<T: Float> crate::op::Op<T> for Conv2D {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // Grab inputs
        let x = &ctx.input(0);
        let w = &ctx.input(1);
        let result = conv2d_impl(
            x,
            w,
            self.pad,
            self.pad,
            self.stride,
            self.stride,
            self.dilation,
            self.dilation,
        );
        match result {
            Ok((y, cols)) => {
                ctx.append_output(y);
                ctx.append_output(cols);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let x = ctx.input(0);
        let w = ctx.input(1);

        let gx = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&w, false)
            .build(super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let cols = nth_tensor(y, 1);
        let gw = Tensor::builder(ctx.graph())
            .append_input(&cols, false)
            .append_input(&gy, false)
            .append_input(&w, false)
            .append_backprop_input(&x)
            .append_backprop_input(&gy)
            .build(Conv2DFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(Some(gw));
    }
}

impl<T: Float> crate::op::Op<T> for Conv2DWithCols {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // Grab inputs
        let cols = &ctx.input(0);
        let w = &ctx.input(1);
        let y = conv2d_with_cols_impl(cols, w);
        ctx.append_output(y);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let cols = ctx.input(0);
        let w = ctx.input(1);
        let y = ctx.output();
        let gy = ctx.output_grad();

        let gx = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&w, false)
            .build(super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let gw = Tensor::builder(ctx.graph())
            .append_input(&cols, false)
            .append_input(&gy, false)
            .append_input(&w, false)
            .append_backprop_input(&y.get_backprop_input(0))
            .append_backprop_input(&gy)
            .build(Conv2DFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(Some(gw));
    }
}

fn conv2d_filter_grad_impl<F: Float>(
    cols: &NdArrayView<F>,
    gy: &NdArrayView<F>,
    w: &NdArrayView<F>,
) -> NdArray<F> {
    let k_shape = w.shape();
    let cols_shape = cols.shape();
    let gy_shape = gy.shape();

    let size_per_batch_g = { gy_shape[1] * gy_shape[2] * gy_shape[3] };
    let size_per_batch_c =
        { cols_shape[1] * cols_shape[2] * cols_shape[3] * cols_shape[4] * cols_shape[5] };
    let (xch, kh, kw) = (k_shape[1], k_shape[2], k_shape[3]);
    let (batch_size, ych, yh, yw) = (gy_shape[0], gy_shape[1], gy_shape[2], gy_shape[3]);

    let cols = cols.as_ptr();

    let copied_gy;
    let gy_ptr = if gy.is_standard_layout() {
        gy.as_ptr()
    } else {
        copied_gy = ndarray_ext::deep_copy(gy);
        copied_gy.as_ptr()
    };

    let gw_len = ych * xch * kh * kw;
    let mut gw = Vec::with_capacity(gw_len);
    let gw_head: *mut F = gw.as_mut_ptr();

    #[cfg(feature = "blas")]
    {
        let m = ych as BlasIF;
        let n = (kh * kw * xch) as BlasIF;
        let k = (yh * yw) as BlasIF;
        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<$ty, F>() {
                    for i in 0..batch_size {
                        unsafe {
                            $f(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasTrans,
                                m,
                                n,
                                k,
                                1.,
                                gy_ptr.add(i * size_per_batch_g) as *const $ty,
                                k,
                                cols.add(i * size_per_batch_c) as *const $ty,
                                k,
                                if i == 0 { 0. } else { 1. },
                                gw_head as *mut $ty,
                                n,
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
                if crate::same_type::<$ty, F>() {
                    for i in 0..batch_size {
                        unsafe {
                            matrixmultiply::$f(
                                m,
                                k,
                                n,
                                1.,                                             // alpha
                                gy_ptr.add(i * size_per_batch_g) as *const $ty, // a
                                rsa as isize,
                                csa as isize,
                                cols.add(i * size_per_batch_c) as *const $ty, // b
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
        gw.set_len(gw_len);
        NdArray::from_shape_vec_unchecked(k_shape, gw)
    }
}

impl<T: Float> crate::op::Op<T> for Conv2DFilterGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let cols = &ctx.input(0); // must be columns
        let gy = &ctx.input(1);
        let w = &ctx.input(2);
        let gw = conv2d_filter_grad_impl(cols, gy, w);
        ctx.append_output(gw);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let cols = ctx.input(0);
        let gy = ctx.input(1); // For example, gradient of output of Conv2D.
        let ggw = ctx.output_grad();
        let y = ctx.output();

        // grad grad
        let gx = Tensor::builder(ctx.graph())
            .append_input(&gy, false)
            .append_input(&ggw, false)
            .build(super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        let ggy = Tensor::builder(ctx.graph())
            .append_input(&cols, false)
            .append_input(&ggw, false)
            .append_backprop_input(&y.get_backprop_input(0))
            .append_backprop_input(&ggw)
            .build(Conv2DWithCols {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(Some(ggy));
    }
}
