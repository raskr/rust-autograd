use super::*;
use crate::NdArray;
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

impl<T: Float> crate::op::Op<T> for Conv2D {
    fn name(&self) -> &str {
        "Conv2D"
    }

    #[allow(unused_mut)]
    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> crate::op::ComputeResults<'v, T> {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let x = &xs[0];
        let w = &xs[1];

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
                "ag::conv2d: Number of input's channel ({:?}) must match filter's second dim ({:?})",
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
        let x_len = x.len();

        // is input dirty?
        let copied_x = ndarray_ext::copy_if_dirty(x);
        let copied_w = ndarray_ext::copy_if_dirty(w);

        // Prepare pointers to buffers
        let x_p = copied_x.map(|inner| inner.as_ptr()).unwrap_or(x.as_ptr());
        let w_p = copied_w.map(|inner| inner.as_ptr()).unwrap_or(w.as_ptr());
        let x_p = unsafe { slice::from_raw_parts(x_p, x_len) };

        let mut y = uninitialized_vec(batch_size * size_per_batch_y);

        let c = im2col_batch(
            x_p,
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
            if same_type::<T, f32>() {
                crate::ops::dot_ops::cblas_sgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w_p as *const f32; batch_size],
                    crate::dot_ops::get_region_heads(batch_size, c.as_ptr(), c.len()),
                    &[0.],
                    crate::dot_ops::get_region_heads(batch_size, y.as_ptr(), y.len()),
                    1,
                    batch_size,
                );
            } else if same_type::<T, f64>() {
                crate::ops::dot_ops::cblas_dgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w_p as *const f64; batch_size],
                    crate::dot_ops::get_region_heads(batch_size, c.as_ptr(), c.len()),
                    &[0.],
                    crate::dot_ops::get_region_heads(batch_size, y.as_ptr(), y.len()),
                    1,
                    batch_size,
                );
            } else {
                panic!("gemm supports only f32 and f64.")
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            let w_len = w.len();
            let w_slice = unsafe { slice::from_raw_parts(w_p, w_len) };
            let size_per_batch_c = xch * kw * kh * yh * yw;
            (0..batch_size).into_par_iter().for_each(|i| {
                // for each batch
                unsafe {
                    let c_region_head: *const T = &c[i * size_per_batch_c];
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
                            w_slice.as_ptr() as *const f32,
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
                            w_slice.as_ptr() as *const f64,
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

        vec![
            Ok(crate::ArrRepr::Owned(y)),
            Ok(crate::ArrRepr::Owned(cols)),
        ]
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

        let cols = &crate::ops::nth_tensor(y, 1);
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

impl<T: Float> crate::op::Op<T> for Conv2DWithCols {
    fn name(&self) -> &str {
        "Conv2DWithCols"
    }

    #[allow(unused_mut)]
    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> crate::op::ComputeResults<'v, T> {
        // Grab inputs
        let xs = ctx.grab_inputs();
        let cols = &xs[0];
        let w = &xs[1];

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
        let copied_w = ndarray_ext::copy_if_dirty(w);
        let w_ptr = copied_w.map(|inner| inner.as_ptr()).unwrap_or(w.as_ptr());
        let mut y = uninitialized_vec(batch_size * size_per_batch_y);

        #[cfg(feature = "mkl")]
        {
            if same_type::<T, f32>() {
                crate::ops::dot_ops::cblas_sgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w_ptr as *const f32; batch_size],
                    crate::dot_ops::get_region_heads(batch_size, cols.as_ptr(), cols.len()),
                    &[0.],
                    crate::dot_ops::get_region_heads(batch_size, y.as_ptr(), y.len()),
                    1,
                    batch_size,
                );
            } else {
                crate::ops::dot_ops::cblas_dgemm_batch_wrapper(
                    false,
                    false,
                    m,
                    n,
                    k,
                    &[1.],
                    vec![w_ptr as *const f64; batch_size],
                    crate::dot_ops::get_region_heads(batch_size, cols.as_ptr(), cols.len()),
                    &[0.],
                    crate::dot_ops::get_region_heads(batch_size, y.as_ptr(), y.len()),
                    1,
                    batch_size,
                );
            }
        }
        #[cfg(not(feature = "mkl"))]
        {
            let c_slice = unsafe { slice::from_raw_parts(cols, cols.len()) };
            let w_slice = unsafe { slice::from_raw_parts(w_ptr, w.len()) };
            let size_per_batch_c =
                { cols_shape[1] * cols_shape[2] * cols_shape[3] * cols_shape[4] * cols_shape[5] };
            // fallback: parallel sgemm using rayon
            (0..batch_size).into_par_iter().for_each(|i| unsafe {
                let c_region_head =
                    c_slice.as_ptr().offset((i * size_per_batch_c) as isize) as *const T;
                let y_region_head: *mut T = mem::transmute(&y[i * size_per_batch_y]);
                slow_gemm!(
                    false,
                    false,
                    w_slice.as_ptr(),
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

        vec![Ok(crate::ArrRepr::Owned(y))]
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

impl<T: Float> crate::op::Op<T> for Conv2DFilterGrad {
    fn name(&self) -> &str {
        "Conv2DFilterGrad"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> crate::op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let cols = &xs[0]; // must be columns
        let gy = &xs[1];

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

        let cols = cols.as_ptr();
        let copied_gy = ndarray_ext::copy_if_dirty(gy);
        let gy = copied_gy.map(|inner| inner.as_ptr()).unwrap_or(gy.as_ptr());

        let mut gw = uninitialized_vec::<T>(ych * xch * kh * kw);
        let gw_head: *mut T = gw.as_mut_ptr();

        for i in 0..batch_size {
            #[cfg(feature = "mkl")]
            unsafe {
                if same_type::<T, f32>() {
                    crate::dot_ops::cblas_sgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        gy.offset((i * size_per_batch_g) as isize) as *const f32,
                        cols.offset((i * size_per_batch_c) as isize) as *const f32,
                        if i == 0 { 0. } else { 1. },
                        gw_head as *mut f32,
                    );
                } else if same_type::<T, f64>() {
                    crate::dot_ops::cblas_dgemm_wrapper(
                        false,
                        true,
                        m,
                        n,
                        k,
                        1.,
                        gy.offset((i * size_per_batch_g) as isize) as *const f64,
                        cols.offset((i * size_per_batch_c) as isize) as *const f64,
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
                        gy.offset((i * size_per_batch_g) as isize) as *const f32,
                        cols.offset((i * size_per_batch_c) as isize) as *const f32,
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
        vec![Ok(crate::ArrRepr::Owned(
            NdArray::from_shape_vec(k_shape, gw).unwrap(),
        ))]
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
    use crate::op::Op;
    let op = Conv2D {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let x = ndarray::Array1::range(0., 2. * 2. * 3. * 3., 1.)
        .into_shape((2, 2, 3, 3))
        .unwrap()
        .into_dyn();

    let w = crate::ndarray_ext::ones(&[
        /*out_ch=*/ 2, /*in_ch=*/ 2, /*row=*/ 2, /*col=*/ 2,
    ]);

    let y = op.compute(crate::runtime::OpComputeContext::new(
        vec![crate::zeros(&[1])], // dummy
        vec![x.view(), w.view()],
    ));

    assert_eq!(
        y[0].as_ref().unwrap().to_owned().as_slice().unwrap(),
        &[52., 60., 76., 84., 52., 60., 76., 84., 196., 204., 220., 228., 196., 204., 220., 228.,]
    );
}
