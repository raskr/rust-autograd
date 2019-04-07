use super::*;
use std::slice;
use NdArray;

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

#[cfg(not(feature = "mkl"))]
macro_rules! slow_gemm {
    ($trans_a:expr, $trans_b:expr, $a:expr, $b:expr, $c:expr,
        $m:expr, $n:expr, $k:expr, $alpha:expr, $beta:expr) => {
        let rsa = if $trans_a { 1 } else { $k };
        let csa = if $trans_a { $m } else { 1 };
        let rsb = if $trans_b { 1 } else { $n };
        let csb = if $trans_b { $k } else { 1 };
        let rsc = $n;
        let csc = 1;
        if same_type::<T, f32>() {
            matrixmultiply::sgemm(
                $m,
                $k,
                $n,
                $alpha as f32,
                $a as *const f32,
                rsa as isize,
                csa as isize,
                $b as *const f32,
                rsb as isize,
                csb as isize,
                $beta as f32,
                $c as *mut f32,
                rsc as isize,
                csc as isize,
            )
        } else if same_type::<T, f64>() {
            matrixmultiply::dgemm(
                $m,
                $k,
                $n,
                $alpha,
                $a as *const f64,
                rsa as isize,
                csa as isize,
                $b as *const f64,
                rsb as isize,
                csb as isize,
                $beta,
                $c as *mut f64,
                rsc as isize,
                csc as isize,
            )
        } else {
            panic!("matrixmultiply::?gemm supports only f32 and f64.")
        }
    };
}

impl<T: Float> ::op::Op<T> for Conv2D {
    fn name(&self) -> &str {
        "Conv2D"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];
        let w: &NdArray<T> = xs[1];

        // Extract size params
        let (batch_size, xch, xh, xw) = {
            let x_shape = x.shape();
            assert_eq!(
                x_shape.len(),
                4,
                "ag::conv2d: Input must be 4D (got {:?})",
                x_shape
            );
            (x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        };
        let (ych, kh, kw) = {
            let k_shape = w.shape();
            assert_eq!(
                k_shape.len(),
                4,
                "ag::conv2d: filter must be 4D (got {:?})",
                k_shape
            );
            assert_eq!(
                xch, k_shape[1],
                "ag::conv2d: Number of input's channel ({:?}) must match second filter dim ({:?})",
                xch, k_shape[1]
            );
            (k_shape[0], k_shape[2], k_shape[3])
        };
        let yh = get_yh!(self, xh, kh);
        let yw = get_yw!(self, xw, kw);

        let size_per_batch_y = ych * yh * yw;

        // Parameters for sgemm
        let m = ych;
        let n = yh * yw;
        let k = xch * kh * kw;

        // Prepare pointers to buffers
        let x = unsafe { slice::from_raw_parts(x.as_ptr(), batch_size * xch * xh * xw) };
        let y = ::dot_ops::uninitialized_vec(batch_size * size_per_batch_y);

        let c = im2col_batch(
            x,
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

        #[cfg(feature = "mkl")]
        {
            let w: *const T = w.as_ptr();
            if same_type::<T, f32>() {
                ::ops::dot_ops::cblas_sgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f32; batch_size],
                    ::dot_ops::get_region_heads(batch_size, c.as_slice()),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, y.as_slice()),
                    1,
                    batch_size,
                );
            } else if same_type::<T, f64>() {
                ::ops::dot_ops::cblas_dgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f64; batch_size],
                    ::dot_ops::get_region_heads(batch_size, c.as_slice()),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, y.as_slice()),
                    1,
                    batch_size,
                );
            } else {
                panic!("gemm supports only f32 and f64.")
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            let size_per_batch_c = xch * kw * kh * yh * yw;
            (0..batch_size).into_par_iter().for_each(|i| {
                // for each batch
                unsafe {
                    let w: *const T = mem::transmute(w.as_ptr());
                    let c_region_head = &c[i * size_per_batch_c] as *const T;
                    let y_region_head: *mut T = mem::transmute(&y[i * size_per_batch_y]);
                    let trans_a = false;
                    let trans_b = false;
                    let rsa = if trans_a { 1 } else { k };
                    let csa = if trans_a { m } else { 1 };
                    let rsb = if trans_b { 1 } else { n };
                    let csb = if trans_b { k } else { 1 };
                    let rsc = n;
                    let csc = 1;
                    if same_type::<T, f32>() {
                        matrixmultiply::sgemm(
                            m,
                            k,
                            n,
                            1.,
                            w as *const f32,
                            rsa as isize,
                            csa as isize,
                            c_region_head as *const f32,
                            rsb as isize,
                            csb as isize,
                            0.,
                            y_region_head as *mut f32,
                            rsc as isize,
                            csc as isize,
                        );
                    } else if same_type::<T, f64>() {
                        matrixmultiply::dgemm(
                            m,
                            k,
                            n,
                            1.,
                            w as *const f64,
                            rsa as isize,
                            csa as isize,
                            c_region_head as *const f64,
                            rsb as isize,
                            csb as isize,
                            0.,
                            y_region_head as *mut f64,
                            rsc as isize,
                            csc as isize,
                        );
                    } else {
                        panic!("gemm supports only f32 and f64.")
                    }
                }
            });
        }
        // move vectors into ndarrays
        let y = NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, ych, yh, yw]), y).unwrap();

        let cols =
            NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, xch, kw, kh, yh, yw]), c).unwrap();

        vec![Ok(y), Ok(cols)]
    }

    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = xs[0];
        let w = xs[1];

        let gx = Tensor::builder().set_inputs(vec![gy, w]).build(
            super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let cols = &::ops::nth_tensor(y, 1);
        let gw = Tensor::builder()
            .set_inputs(vec![cols, gy, w])
            .set_backprop_inputs(vec![x.clone(), gy.clone()])
            .build(Conv2DFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        vec![Some(gx), Some(gw)]
    }
}

impl<T: Float> ::op::Op<T> for Conv2DWithCols {
    fn name(&self) -> &str {
        "Conv2DWithCols"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let cols: &NdArray<T> = xs[0];
        let w: &NdArray<T> = xs[1];

        // Extract size params
        let cols_shape = cols.shape();
        let k_shape = w.shape();
        let (ych, xch, kh, kw) = { (k_shape[0], k_shape[1], k_shape[2], k_shape[3]) };
        let yh = cols_shape[4];
        let yw = cols_shape[5];
        let batch_size = cols_shape[0];

        // Parameters for sgemm
        let size_per_batch_y = ych * yh * yw;
        let m = ych;
        let n = yh * yw;
        let k = xch * kh * kw;

        // Prepare buffers
        let c = unsafe { slice::from_raw_parts(cols.as_ptr(), cols.len()) };
        let y = ::dot_ops::uninitialized_vec(batch_size * size_per_batch_y);

        #[cfg(feature = "mkl")]
        {
            let w: *const T = w.as_ptr();
            if same_type::<T, f32>() {
                ::ops::dot_ops::cblas_sgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f32; batch_size],
                    ::dot_ops::get_region_heads(batch_size, c),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, y.as_slice()),
                    1,
                    batch_size,
                );
            } else {
                ::ops::dot_ops::cblas_dgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w as *const f64; batch_size],
                    ::dot_ops::get_region_heads(batch_size, c),
                    &[0.],
                    ::dot_ops::get_region_heads(batch_size, y.as_slice()),
                    1,
                    batch_size,
                );
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            let size_per_batch_c =
                { cols_shape[1] * cols_shape[2] * cols_shape[3] * cols_shape[4] * cols_shape[5] };
            // fallback: parallel sgemm using rayon
            (0..batch_size).into_par_iter().for_each(|i| unsafe {
                let c_region_head = &c[i * size_per_batch_c] as *const T;
                let y_region_head: *mut T = mem::transmute(&y[i * size_per_batch_y]);
                let w: *const T = mem::transmute(w.as_ptr());
                slow_gemm!(
                    false,
                    false,
                    w,
                    c_region_head,
                    y_region_head,
                    m,
                    n,
                    k,
                    1.,
                    0.
                );
            });
        }
        // move vectors into ndarrays
        let y = NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, ych, yh, yw]), y).unwrap();

        vec![Ok(y)]
    }

    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let cols = xs[0];
        let w = xs[1];

        let gx = Tensor::builder().set_inputs(vec![gy, w]).build(
            super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let gw = Tensor::builder()
            .set_inputs(vec![cols, gy, w])
            .set_backprop_inputs(vec![
                y.inputs_on_backprop.as_ref().unwrap()[0].clone(),
                gy.clone(),
            ])
            .build(Conv2DFilterGrad {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        vec![Some(gx), Some(gw)]
    }
}

impl<T: Float> ::op::Op<T> for Conv2DFilterGrad {
    fn name(&self) -> &str {
        "Conv2DFilterGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let cols = xs[0]; // must be columns
        let gy = xs[1];
        let k_shape = xs[2].shape();
        let cols_shape = cols.shape();
        let gy_shape = gy.shape();

        let size_per_batch_g = { gy_shape[1] * gy_shape[2] * gy_shape[3] };

        let size_per_batch_c =
            { cols_shape[1] * cols_shape[2] * cols_shape[3] * cols_shape[4] * cols_shape[5] };

        let (xch, kh, kw) = (k_shape[1], k_shape[2], k_shape[3]);
        let (batch_size, ych, yh, yw) = (gy_shape[0], gy_shape[1], gy_shape[2], gy_shape[3]);

        let m = ych;
        let n = kh * kw * xch;
        let k = yh * yw;

        // Prepare bufs
        let cols = unsafe { slice::from_raw_parts(cols.as_ptr(), cols.len()) };
        let gy = unsafe { slice::from_raw_parts(gy.as_ptr(), gy.len()) };
        let mut gw = ::dot_ops::uninitialized_vec::<T>(ych * xch * kh * kw);
        let gw_head: *mut T = gw.as_mut_ptr();

        for i in 0..batch_size {
            #[cfg(feature = "mkl")]
            {
                if same_type::<T, f32>() {
                    ::dot_ops::cblas_sgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        &gy[i * size_per_batch_g] as *const T as *const f32,
                        &cols[i * size_per_batch_c] as *const T as *const f32,
                        if i == 0 { 0. } else { 1. },
                        gw_head as *mut f32,
                    );
                } else if same_type::<T, f64>() {
                    ::dot_ops::cblas_dgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        &gy[i * size_per_batch_g] as *const T as *const f64,
                        &cols[i * size_per_batch_c] as *const T as *const f64,
                        if i == 0 { 0. } else { 1. },
                        gw_head as *mut f64,
                    );
                } else {
                    panic!("gemm supports only f32 and f64.")
                }
            }
            #[cfg(not(feature = "mkl"))]
            {
                unsafe {
                    slow_gemm!(
                        false,
                        true,
                        &gy[i * size_per_batch_g] as *const T as *const f32,
                        &cols[i * size_per_batch_c] as *const T as *const f32,
                        gw_head as *mut f32,
                        m,
                        n,
                        k,
                        1.,
                        if i == 0 { 0. } else { 1. }
                    );
                }
            }
        }
        vec![Ok(NdArray::from_shape_vec(k_shape, gw).unwrap())]
    }

    fn grad(&self, ggw: &Tensor<T>, xs: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let cols = xs[0];
        let gy = xs[1]; // For example, gradient of output of Conv2D.

        // grad grad
        let gx = Tensor::builder().set_inputs(vec![gy, ggw]).build(
            super::conv2d_transpose::Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let ggy = Tensor::builder()
            .set_inputs(vec![cols, ggw])
            .set_backprop_inputs(vec![
                y.inputs_on_backprop.as_ref().unwrap()[0].clone(),
                ggw.clone(),
            ])
            .build(Conv2DWithCols {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            });

        vec![Some(gx), Some(ggy), None]
    }
}

#[test]
fn test_tensor_size_after_convolution() {
    let op = Conv2D {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);
    let yh = get_yh!(&op, xh, kh);
    let yw = get_yw!(&op, xw, kw);
    assert_eq!(yh, 2);
    assert_eq!(yw, 2);
}

#[test]
fn test_conv2d() {
    use op::Op;
    let op = Conv2D {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let x = ndarray::Array1::range(0., 2. * 2. * 3. * 3., 1.)
        .into_shape((2, 2, 3, 3))
        .unwrap()
        .into_dyn();

    let w = ::ndarray_ext::ones(&[
        /*out_ch=*/ 2, /*in_ch=*/ 2, /*row=*/ 2, /*col=*/ 2,
    ]);

    let y = op.compute(::runtime::OpComputeContext::new(
        &::ops::zeros(&[0]), // dummy (not used)
        vec![&x, &w],
    ));

    assert_eq!(
        y[0].as_ref().unwrap().as_slice().unwrap(),
        &[52., 60., 76., 84., 52., 60., 76., 84., 196., 204., 220., 228., 196., 204., 220., 228.,]
    );
}
